# Generative AI Integration in Unreal Engine

This project was developed as part of a master's thesis.
It demonstrates the integration of generative AI tools
into the Unreal Engine development workflow.

## Project Structure

- `backend/` – Python backend and AI model integration.
- `unreal-project/` – Unreal Engine project.
- `docs/` – Additional documentation.

## Requirements

- Unreal Engine 5.5
- Python 3.12
- Git
- Git LFS

## Backend Setup

1. Install Python 3.12.
2. Navigate to the backend directory.
3. Create a virtual environment.
4. Install dependencies:

   `pip install -r requirements.txt`

5. Configure required environment variables.
6. Start the backend server using the provided startup script.

## Unreal Engine Setup

1. Open the `.uproject` file inside `unreal-project/`.
2. Use Unreal Engine 5.5.
3. Allow Unreal Engine to generate required files.
4. Configure the backend connection.
5. Start the backend before using AI functionality.

## License

This project is licensed under the MIT License.
See the LICENSE file for details.

## LLM Server Structure

The server was refactored from one file into a small package:

- `llm_server.py` - thin entrypoint (starts Uvicorn)
- `llm_backend/app.py` - creates FastAPI app and wires routes
- `llm_backend/routes.py` - HTTP endpoints (`/generate`, `/result/{id}`, `/generate_image`, `/image_result/{id}`, `/edit_image`, `/image_edit_result/{id}`, `/generate_music`, `/music_result/{id}`, `/health`)
- `llm_backend/schemas.py` - Pydantic request/response models
- `llm_backend/runtime.py` - shared runtime state (model/tokenizer loading, device, executor, task stores, image pipeline loader, image edit pipeline loader, music model loader)
- `llm_backend/config.py` - constants and environment/output path setup
- `llm_backend/services/text_service.py` - text generation logic
- `llm_backend/services/image_service.py` - image generation and save logic
- `llm_backend/services/image_edit_service.py` - image editing and save logic

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
- `image_edit_pipeline`
- `music_pipeline`
