from pathlib import Path
from typing import Optional

from PIL import Image

from .. import runtime
from ..config import OUTPUT_DIR

from llm_backend.services import context_service
from llm_backend.services.text_service import _summarizer_generate_fn


def _build_image_prompt(user_request: str) -> str:
    """Combine the game's design summary and earlier prompts with the new request to produce
    a single, self-contained prompt suitable for the image pipeline."""
    context = context_service.load_context()
    summary = context["summary"].strip()
    recent_turns = context_service.build_recent_turns_prompt(context).strip()

    if not summary and not recent_turns:
        return user_request

    recent_section = f"\n\n{recent_turns}\n" if recent_turns else ""

    distill_prompt = f"""Game design document:
{summary}
{recent_section}
New image request: {user_request}

Rewrite the new image request as a single, self-contained, descriptive image-generation prompt.
Use the game design document and the earlier project prompts only as background context for style,
setting, and continuity. Treat earlier prompts as historical references, not as the current request.
Incorporate only details that are clearly relevant to this specific image. Do not invent unrelated details.

STRICT LENGTH LIMIT: maximum 60 words. Prioritize concrete visual details (subject, pose,
clothing, colors, setting, lighting, art style) over narrative/emotional description.
Output ONLY the final prompt text, nothing else — no explanations, no quotation marks.
"""
    result = _summarizer_generate_fn(distill_prompt).strip()
    print(result)
    return _clip_safe_truncate(result)


def _clip_safe_truncate(text: str, max_words: int = 60) -> str:
    """Hard safety net in case the model ignores the word limit — CLIP truncates at 77 tokens,
    so keep well under that since words often split into multiple tokens."""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def _apply_pixel_art_postprocess(image: Image.Image, downscale_factor: int = 8) -> Image.Image:
    """Downscale in blocks using nearest-neighbor sampling to preserve crisp pixel-art edges."""
    if downscale_factor <= 1:
        return image

    width, height = image.size
    reduced_width = max(1, width // downscale_factor)
    reduced_height = max(1, height // downscale_factor)
    reduced = image.resize((reduced_width, reduced_height), resample=Image.Resampling.NEAREST)
    return reduced.resize((width, height), resample=Image.Resampling.NEAREST)


def generate_image(task_id: str, prompt: str, name: Optional[str], height: int, width: int) -> None:
    try:
        if runtime.image_pipe is None:
            raise RuntimeError("Image pipeline is not loaded yet")

        final_prompt = _build_image_prompt(prompt)
        print(f"Generating image prompt: {final_prompt}")

        image = runtime.image_pipe(final_prompt, height=height, width=width).images[0]
        image = _apply_pixel_art_postprocess(image)

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