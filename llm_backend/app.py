from fastapi import FastAPI, Request
from fastapi.responses import Response

from .routes import router
from .runtime import start_background_loaders

app = FastAPI()
app.include_router(router)


@app.middleware("http")
async def print_response_body(request: Request, call_next):
	response = await call_next(request)

	body = b""
	async for chunk in response.body_iterator:
		body += chunk

	try:
		body_text = body.decode(response.charset or "utf-8")
	except UnicodeDecodeError:
		body_text = body.decode("utf-8", errors="replace")

	print(f"{request.method} {request.url.path} -> {body_text}")

	headers = dict(response.headers)
	headers.pop("content-length", None)

	return Response(
		content=body,
		status_code=response.status_code,
		headers=headers,
		media_type=response.media_type,
	)

start_background_loaders()
