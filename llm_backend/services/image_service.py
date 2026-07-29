from pathlib import Path
from typing import Optional

from .. import runtime
from ..config import OUTPUT_DIR

from llm_backend.services import context_service
from llm_backend.services.text_service import _summarizer_generate_fn


def _build_image_prompt(user_request: str) -> str:
    """Combine the game's design summary with the new request to produce
    a single, self-contained prompt suitable for the image pipeline."""
    context = context_service.load_context()
    summary = context["summary"].strip()

    if not summary:
        return user_request

    distill_prompt = f"""Game design document:
{summary}

New image request: {user_request}

Rewrite the new image request as a single, self-contained, descriptive image-generation prompt.
Incorporate relevant details from the design document (art style, characters, setting, tone)
only if clearly relevant to this specific request. Do not invent unrelated details.
Output ONLY the final prompt text, nothing else.
"""
    return _summarizer_generate_fn(distill_prompt).strip()


def generate_image(task_id: str, prompt: str, name: Optional[str], height: int, width: int) -> None:
    try:
        if runtime.image_pipe is None:
            raise RuntimeError("Image pipeline is not loaded yet")

        final_prompt = _build_image_prompt(prompt)
        print(f"Generating image prompt: {final_prompt}")

        image = runtime.image_pipe(final_prompt, height=height, width=width).images[0]

        filename = name or f"{task_id}.png"
        candidate = Path(filename)
        if candidate.suffix == "":
            candidate = Path(f"{candidate.name}.png")

        out_path = OUTPUT_DIR / candidate.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")

        runtime.image_results_store[task_id] = str(out_path)

        context_service.update_context_with_new_turn(
            user_message=prompt,
            assistant_response="image",
            generate_fn=_summarizer_generate_fn,
        )

    except Exception as error:
        runtime.image_results_store[task_id] = f"Error: {error}"