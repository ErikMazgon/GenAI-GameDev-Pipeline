import uvicorn
from llm_backend.app import app
from llm_backend.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)