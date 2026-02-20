from fastapi import FastAPI

from .routes import router
from .runtime import start_background_loaders

app = FastAPI()
app.include_router(router)

start_background_loaders()
