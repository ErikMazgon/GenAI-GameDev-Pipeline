from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, configure_cache_paths, ensure_output_dir

configure_cache_paths()
ensure_output_dir()

print("Nalaganje modela...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="cuda" if device == "cuda" else "auto",
)

print(f"Model device: {next(model.parameters()).device}")
print("Model naložen!")

executor = ThreadPoolExecutor(max_workers=2)
results_store: Dict[str, Optional[str]] = {}
image_results_store: Dict[str, Optional[str]] = {}

image_pipe = None
image_load_status: str = "not_started"
image_load_error: Optional[str] = None


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
            device_map="cuda",
        )
        image_pipe.load_lora_weights("nerijs/pixel-art-xl")
        image_load_status = "loaded"
        print("Image pipeline loaded!")
    except Exception as error:
        image_pipe = None
        image_load_status = "error"
        image_load_error = str(error)
        print(f"Failed to load image pipeline: {error}")


def start_background_loaders() -> None:
    executor.submit(load_image_pipeline)
