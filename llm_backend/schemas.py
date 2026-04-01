from typing import Optional

from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    task_id: str
    status: str


class ResultResponse(BaseModel):
    status: str
    response: Optional[str] = None


class ImagePromptRequest(BaseModel):
    prompt: str
    name: Optional[str] = None
    height: int = 1024
    width: int = 1024


class MusicPromptRequest(BaseModel):
    prompt: str
    name: Optional[str] = None
    max_new_tokens: int = 256
