from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, MusicgenForConditionalGeneration, BitsAndBytesConfig

from .config import MODEL_NAME, MUSIC_MODEL_NAME, configure_cache_paths, ensure_output_dir

configure_cache_paths()
ensure_output_dir()

print("Nalaganje modela...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="cuda" if device == "cuda" else "auto",
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print(f"Model device: {next(model.parameters()).device}")
print("Model naložen!")

executor = ThreadPoolExecutor(max_workers=2)
results_store: Dict[str, Optional[str]] = {}
image_results_store: Dict[str, Optional[str]] = {}
image_edit_results_store: Dict[str, Optional[str]] = {}
music_results_store: Dict[str, Optional[str]] = {}

image_pipe = None
image_load_status: str = "not_started"
image_load_error: Optional[str] = None

image_edit_pipe = None
image_edit_load_status: str = "not_started"
image_edit_load_error: Optional[str] = None

music_processor = None
music_model = None
music_load_status: str = "not_started"
music_load_error: Optional[str] = None


def load_image_pipeline() -> None:
    global image_pipe, image_load_status, image_load_error
    try:
        from diffusers import DiffusionPipeline

        image_load_status = "loading"
        image_load_error = None
        print("Loading image pipeline (background)...")
        image_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
        )
        image_pipe.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
        if hasattr(image_pipe, "vae"):
            image_pipe.vae.to(dtype=torch.float16 if device == "cuda" else torch.float32)
        image_pipe.load_lora_weights("nerijs/pixel-art-xl")
        image_load_status = "loaded"
        print("Image pipeline loaded!")
    except Exception as error:
        image_pipe = None
        image_load_status = "error"
        image_load_error = str(error)
        print(f"Failed to load image pipeline: {error}")
    finally:
        executor.submit(load_music_pipeline)
        executor.submit(load_image_edit_pipeline)


def load_music_pipeline() -> None:
    global music_processor, music_model, music_load_status, music_load_error
    try:
        music_load_status = "loading"
        music_load_error = None
        print("Loading music model (background)...")
        music_processor = AutoProcessor.from_pretrained(MUSIC_MODEL_NAME)
        music_model = MusicgenForConditionalGeneration.from_pretrained(
            MUSIC_MODEL_NAME,
            attn_implementation="eager",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        music_model = music_model.to(device)
        music_load_status = "loaded"
        print("Music model loaded!")

    except Exception as error:
        music_processor = None
        music_model = None
        music_load_status = "error"
        music_load_error = str(error)
        print(f"Failed to load music model: {error}")



def load_image_edit_pipeline() -> None:
    global image_edit_pipe, image_edit_load_status, image_edit_load_error
    try:
        from diffusers import LCMScheduler, StableDiffusionXLImg2ImgPipeline

        if image_pipe is None:
            raise RuntimeError("Image pipeline must load before image edit pipeline")

        image_edit_load_status = "loading"
        image_edit_load_error = None
        print("Building shared image edit pipeline (background)...")
        image_edit_pipe = StableDiffusionXLImg2ImgPipeline(**image_pipe.components)
        image_edit_pipe.scheduler = LCMScheduler.from_config(image_edit_pipe.scheduler.config)
        image_edit_pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lora")
        image_edit_pipe.load_lora_weights("nerijs/pixel-art-xl", adapter_name="pixel")
        image_edit_pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
        image_edit_pipe = image_edit_pipe.to(device)
        image_edit_load_status = "loaded"
        print("Image edit pipeline loaded!")
        if image_edit_load_status == "loaded" and image_load_status == "loaded" and music_load_status == "loaded":
            print("All models loaded!")
    except Exception as error:
        image_edit_pipe = None
        image_edit_load_status = "error"
        image_edit_load_error = str(error)
        print(f"Failed to load image edit pipeline: {error}")



def start_background_loaders() -> None:
    executor.submit(load_image_pipeline)
