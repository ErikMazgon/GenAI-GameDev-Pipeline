from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import Dict, Optional
from pathlib import Path

# set HF cache before diffusers ever gets imported
os.environ["HF_HOME"] = r"D:\ErikM\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\ErikM\hf_cache\transformers"
os.environ["XDG_CACHE_HOME"] = r"D:\ErikM\hf_cache"

print("Nalaganje modela...")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the TinyLlama model at startup with GPU support
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="cuda" if device == "cuda" else "auto",
)

print(f"Model device: {next(model.parameters()).device}")
print("Model naložen!")

# ---------------------------------------------------------------------------
# Image pipeline – deferred load (runs in background after LLM is ready)
# ---------------------------------------------------------------------------
def _load_image_pipeline():
    """Import diffusers lazily and load the SDXL + LoRA pipeline."""
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
    except Exception as e:
        image_pipe = None
        image_load_status = "error"
        image_load_error = str(e)
        print(f"Failed to load image pipeline: {e}")

# Thread pool for running inference (2 workers: LLM + image)
executor = ThreadPoolExecutor(max_workers=2)

# Store for async results (text)
results_store: Dict[str, Optional[str]] = {}

# Store for async results (image paths)
image_results_store: Dict[str, Optional[str]] = {}

# Image pipeline – will be populated by background loader
image_pipe = None

# NEW: status tracking for image loader
image_load_status: str = "not_started"   # "not_started" | "loading" | "loaded" | "error"
image_load_error: Optional[str] = None

# Kick off image-pipeline load *after* the LLM has finished loading
executor.submit(_load_image_pipeline)

# Define the expected input structure
class PromptRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    task_id: str
    status: str

class ResultResponse(BaseModel):
    status: str
    response: Optional[str] = None

app = FastAPI()

def generate_text(task_id: str, prompt: str):
    """Run model inference in a separate thread"""
    try:
        # Format the prompt as a chat message
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Tokenize and prepare inputs
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )
        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        
        # Store the result
        results_store[task_id] = result.strip()
    except Exception as e:
        results_store[task_id] = f"Error: {str(e)}"

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(r"C:\Users\erik\Documents\Unreal Projects\MyProject\Content\Python\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ImagePromptRequest(BaseModel):
    prompt: str
    name: Optional[str] = None
    height: int = 1024
    width: int = 1024

def generate_image(task_id: str, prompt: str, name: Optional[str], height: int, width: int):
    """Run image inference in a separate thread and save to disk."""
    try:
        if image_pipe is None:
            raise RuntimeError("Image pipeline is not loaded yet")

        image = image_pipe(prompt, height=height, width=width).images[0]

        # Ensure filename has an extension; default to .png
        filename = name or f"{task_id}.png"
        p = Path(filename)
        if p.suffix == "":
            p = Path(f"{p.name}.png")

        out_path = OUTPUT_DIR / p.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Specify format to avoid "unknown file extension" errors
        image.save(out_path, format="PNG")

        image_results_store[task_id] = str(out_path)
    except Exception as e:
        image_results_store[task_id] = f"Error: {str(e)}"

# REQUEST AND RESPONSE IMAGE
@app.post("/generate_image", response_model=GenerateResponse)
async def generate_image_endpoint(request: ImagePromptRequest):
    """Start image generation and return a task ID immediately."""
    # If the loader errored, return that info so caller can act (don't just say 'not ready')
    if image_load_status != "loaded":
        return {
            "task_id": "",
            "status": image_load_status if image_load_status != "error" else f"error: {image_load_error}"
        }

    task_id = str(uuid.uuid4())
    image_results_store[task_id] = None  # pending

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, generate_image, task_id,
        request.prompt, request.name, request.height, request.width,
    )
    return {"task_id": task_id, "status": "processing"}

@app.get("/image_result/{task_id}")
async def get_image_result(task_id: str):
    """Poll for image generation result (file path on disk)."""
    if task_id not in image_results_store:
        return {"status": "not_found", "path": None}

    result = image_results_store[task_id]
    if result is None:
        return {"status": "processing", "path": None}

    path = image_results_store.pop(task_id)
    return {"status": "completed", "path": path}

# REQUEST AND RESPONSE TEXT
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: PromptRequest):
    """Start generation and return task ID immediately"""
    task_id = str(uuid.uuid4())
    results_store[task_id] = None  # Mark as pending
    
    # Run inference in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, generate_text, task_id, request.prompt)
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    """Check if generation is complete and get result"""
    if task_id not in results_store:
        return {"status": "not_found", "response": None}
    
    result = results_store[task_id]
    
    if result is None:
        return {"status": "processing", "response": None}
    else:
        # Clean up after returning result
        response = results_store.pop(task_id)
        return {"status": "completed", "response": response}

@app.post("/generate_sync")
async def generate_sync(request: PromptRequest):
    """Synchronous endpoint (original behavior) - use only for testing"""
    prompt = request.prompt
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=32,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True,
    )
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    return {"response": result.strip()}

@app.get("/health")
async def health():
    return {
        "status": "running",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "device": str(device),
        "image_pipeline": image_load_status,
        "image_pipeline_error": image_load_error,
        "pending_text_tasks": sum(1 for v in results_store.values() if v is None),
        "pending_image_tasks": sum(1 for v in image_results_store.values() if v is None),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)