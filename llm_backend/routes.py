import asyncio
import uuid

from fastapi import APIRouter

from . import runtime
from .schemas import (
    GenerateResponse,
    ImagePromptRequest,
    PromptRequest,
    ResultResponse,
    MusicPromptRequest,
)
from .services.image_service import generate_image
from .services.music_service import generate_music
from .services.text_service import generate_text, generate_text_sync

router = APIRouter()


@router.post("/generate_image", response_model=GenerateResponse)
async def generate_image_endpoint(request: ImagePromptRequest):
    if runtime.image_load_status != "loaded":
        return {
            "task_id": "",
            "status": runtime.image_load_status
            if runtime.image_load_status != "error"
            else f"error: {runtime.image_load_error}",
        }

    task_id = str(uuid.uuid4())
    runtime.image_results_store[task_id] = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        runtime.executor,
        generate_image,
        task_id,
        request.prompt,
        request.name,
        request.height,
        request.width,
    )
    return {"task_id": task_id, "status": "processing"}

@router.post("/generate_music", response_model=GenerateResponse)
async def generate_music_endpoint(request: MusicPromptRequest):
    if runtime.music_load_status != "loaded":
        return {
            "task_id": "",
            "status": runtime.music_load_status
            if runtime.music_load_status != "error"
            else f"error: {runtime.music_load_error}",
        }

    task_id = str(uuid.uuid4())
    runtime.music_results_store[task_id] = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        runtime.executor,
        generate_music,
        task_id,
        request.prompt,
        request.name,
        request.max_new_tokens,
    )
    return {"task_id": task_id, "status": "processing"}


@router.get("/music_result/{task_id}")
async def get_music_result(task_id: str):
    if task_id not in runtime.music_results_store:
        return {"status": "not_found", "path": None}

    result = runtime.music_results_store[task_id]
    if result is None:
        return {"status": "processing", "path": None}

    path = runtime.music_results_store.pop(task_id)
    return {"status": "completed", "path": path}


@router.get("/image_result/{task_id}")
async def get_image_result(task_id: str):
    if task_id not in runtime.image_results_store:
        return {"status": "not_found", "path": None}

    result = runtime.image_results_store[task_id]
    if result is None:
        return {"status": "processing", "path": None}

    path = runtime.image_results_store.pop(task_id)
    return {"status": "completed", "path": path}


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: PromptRequest):
    task_id = str(uuid.uuid4())
    runtime.results_store[task_id] = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(runtime.executor, generate_text, task_id, request.prompt)

    return {"task_id": task_id, "status": "processing"}


@router.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    if task_id not in runtime.results_store:
        return {"status": "not_found", "response": None}

    result = runtime.results_store[task_id]
    if result is None:
        return {"status": "processing", "response": None}

    response = runtime.results_store.pop(task_id)
    return {"status": "completed", "response": response}


@router.post("/generate_sync")
async def generate_sync(request: PromptRequest):
    return {"response": generate_text_sync(request.prompt)}


@router.get("/health")
async def health():
    return {
        "status": "running",
        "model": runtime.MODEL_NAME,
        "device": str(runtime.device),
        "image_pipeline": runtime.image_load_status,
        "image_pipeline_error": runtime.image_load_error,
        "music_pipeline": runtime.music_load_status,
        "music_pipeline_error": runtime.music_load_error,
        "pending_text_tasks": sum(1 for value in runtime.results_store.values() if value is None),
        "pending_image_tasks": sum(1 for value in runtime.image_results_store.values() if value is None),
        "pending_music_tasks": sum(1 for value in runtime.music_results_store.values() if value is None),
    }
