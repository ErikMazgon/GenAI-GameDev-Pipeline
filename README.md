# Mag
Master's Thesis in Unreal Engine

## LLM Server Structure

The server was refactored from one file into a small package:

- `llm_server.py` - thin entrypoint (starts Uvicorn)
- `llm_backend/app.py` - creates FastAPI app and wires routes
- `llm_backend/routes.py` - HTTP endpoints (`/generate`, `/result/{id}`, `/generate_image`, `/image_result/{id}`, `/health`)
- `llm_backend/schemas.py` - Pydantic request/response models
- `llm_backend/runtime.py` - shared runtime state (model/tokenizer loading, device, executor, task stores, image pipeline loader)
- `llm_backend/config.py` - constants and environment/output path setup
- `llm_backend/services/text_service.py` - text generation logic
- `llm_backend/services/image_service.py` - image generation and save logic

## Where to extend

- Add or modify API endpoints in `llm_backend/routes.py`
- Put business logic in `llm_backend/services/`
- Add new request/response models in `llm_backend/schemas.py`
- Keep startup/config constants in `llm_backend/config.py`

## Run locally

From `Content/Python`:

- Start server: `python llm_server.py`
- Health check: `http://127.0.0.1:5000/health`
- Testing: `http://127.0.0.1:5000/docs`


Expected health response includes:

- `status`
- `model`
- `device`
- `image_pipeline`
