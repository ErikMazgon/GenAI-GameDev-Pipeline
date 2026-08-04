import asyncio
import uuid

from fastapi import APIRouter, HTTPException

from . import runtime
from .schemas import (
    GenerateResponse,
    ImageEditPromptRequest,
    ImagePromptRequest,
    PromptRequest,
    ResultResponse,
    MusicPromptRequest,
)
from .services.image_edit_service import edit_image
from .services.image_edit_service import resolve_output_path, resolve_source_image_path
from .services.image_service import generate_image
from .services.music_service import generate_music
from .services.text_service import generate_text 
# , generate_text_sync

router = APIRouter()

@router.post("/generate_text", response_model=GenerateResponse)
async def generate_text_endpoint(request: PromptRequest):
    task_id = str(uuid.uuid4())
    runtime.results_store[task_id] = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(runtime.executor, generate_text, task_id, request.prompt)

    response = {"task_id": task_id, "status": "processing"}
    print(response)
    return response


@router.post("/generate_image", response_model=GenerateResponse)
async def generate_image_endpoint(request: ImagePromptRequest):
    if runtime.image_load_status != "loaded":
        response = {
            "task_id": "",
            "status": runtime.image_load_status
            if runtime.image_load_status != "error"
            else f"error: {runtime.image_load_error}",
        }
        print(response)
        return response

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
    response = {"task_id": task_id, "status": "processing"}
    print(response)
    return response


@router.post("/edit_image", response_model=GenerateResponse)
async def edit_image_endpoint(request: ImageEditPromptRequest):
    if runtime.image_edit_load_status != "loaded":
        response = {
            "task_id": "",
            "status": runtime.image_edit_load_status
            if runtime.image_edit_load_status != "error"
            else f"error: {runtime.image_edit_load_error}",
        }
        print(response)
        return response

    try:
        resolve_source_image_path(request.source_image)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        resolve_output_path(request.name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    task_id = str(uuid.uuid4())
    runtime.image_edit_results_store[task_id] = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        runtime.executor,
        edit_image,
        task_id,
        request.source_image,
        request.prompt,
        request.name,
        request.strength,
        request.guidance_scale,
        request.num_inference_steps,
    )
    response = {"task_id": task_id, "status": "processing"}
    print(response)
    return response


@router.post("/generate_music", response_model=GenerateResponse)
async def generate_music_endpoint(request: MusicPromptRequest):
    if runtime.music_load_status != "loaded":
        response = {
            "task_id": "",
            "status": runtime.music_load_status
            if runtime.music_load_status != "error"
            else f"error: {runtime.music_load_error}",
        }
        print(response)
        return response

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
    response = {"task_id": task_id, "status": "processing"}
    print(response)
    return response




@router.get("/text_result/{task_id}", response_model=ResultResponse)
async def get_text_result(task_id: str):
    if task_id not in runtime.results_store:
        response = {"status": "not_found", "response": None}
        print(response)
        return response

    result = runtime.results_store[task_id]
    if result is None:
        response = {"status": "processing", "response": None}
        print(response)
        return response

    response = runtime.results_store.pop(task_id)
    body = {"status": "completed", "response": response}
    print(body)
    return body


@router.get("/image_result/{task_id}")
async def get_image_result(task_id: str):
    if task_id not in runtime.image_results_store:
        response = {"status": "not_found", "path": None}
        print(response)
        return response

    result = runtime.image_results_store[task_id]
    if result is None:
        response = {"status": "processing", "path": None}
        print(response)
        return response

    path = runtime.image_results_store.pop(task_id)
    body = {"status": "completed", "path": path}
    print(body)
    return body


@router.get("/image_edit_result/{task_id}")
async def get_image_edit_result(task_id: str):
    if task_id not in runtime.image_edit_results_store:
        response = {"status": "not_found", "path": None}
        print(response)
        return response

    result = runtime.image_edit_results_store[task_id]
    if result is None:
        response = {"status": "processing", "path": None}
        print(response)
        return response

    path = runtime.image_edit_results_store.pop(task_id)
    body = {"status": "completed", "path": path}
    print(body)
    return body


@router.get("/music_result/{task_id}")
async def get_music_result(task_id: str):
    if task_id not in runtime.music_results_store:
        response = {"status": "not_found", "path": None}
        print(response)
        return response

    result = runtime.music_results_store[task_id]
    if result is None:
        response = {"status": "processing", "path": None}
        print(response)
        return response

    path = runtime.music_results_store.pop(task_id)
    body = {"status": "completed", "path": path}
    print(body)
    return body





# @router.post("/generate_sync")
# async def generate_sync(request: PromptRequest):
#     return {"response": generate_text_sync(request.prompt)}


@router.get("/health")
async def health():
    response = {
        "status": "running",
        "model": runtime.MODEL_NAME,
        "device": str(runtime.device),
        "image_pipeline": runtime.image_load_status,
        "image_pipeline_error": runtime.image_load_error,
        "image_edit_pipeline": runtime.image_edit_load_status,
        "image_edit_pipeline_error": runtime.image_edit_load_error,
        "music_pipeline": runtime.music_load_status,
        "music_pipeline_error": runtime.music_load_error,
        "pending_text_tasks": sum(1 for value in runtime.results_store.values() if value is None),
        "pending_image_tasks": sum(1 for value in runtime.image_results_store.values() if value is None),
        "pending_image_edit_tasks": sum(1 for value in runtime.image_edit_results_store.values() if value is None),
        "pending_music_tasks": sum(1 for value in runtime.music_results_store.values() if value is None),
    }
    print(response)
    return response
