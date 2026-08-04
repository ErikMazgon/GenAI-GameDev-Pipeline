from pathlib import Path
from PIL import Image
from PIL import ImageOps

from .. import runtime
from ..config import OUTPUT_DIR


def resolve_source_image_path(source_image: str) -> Path:
    candidate = Path(source_image)
    if candidate.is_absolute() or candidate.parent != Path(".") or candidate.name != source_image:
        raise ValueError("source_image must be a filename only, not a full path")

    source_path = OUTPUT_DIR / candidate.name
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    return source_path


def _pad_to_multiple_of_eight(image: Image.Image) -> Image.Image:
    width, height = image.size
    padded_width = ((width + 7) // 8) * 8
    padded_height = ((height + 7) // 8) * 8

    if padded_width == width and padded_height == height:
        return image

    pad_left = (padded_width - width) // 2
    pad_top = (padded_height - height) // 2
    pad_right = padded_width - width - pad_left
    pad_bottom = padded_height - height - pad_top
    return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))


def resolve_output_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.parent != Path(".") or candidate.name != name:
        raise ValueError("name must be a filename only, not a full path")

    if candidate.suffix == "":
        candidate = Path(f"{candidate.name}.png")

    return OUTPUT_DIR / candidate.name


def edit_image(
    task_id: str,
    source_image: str,
    prompt: str,
    name: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
) -> None:
    try:
        if runtime.image_edit_pipe is None:
            raise RuntimeError("Image edit pipeline is not loaded yet")

        source_path = resolve_source_image_path(source_image)
        print(f"Editing image from: {source_path.name}")

        image = Image.open(source_path).convert("RGB")
        image = _pad_to_multiple_of_eight(image)

        # Img2img re-generates the image from the prompt + source image + noise,
        # so this prompt should describe the whole desired scene, not just an edit instruction.
        edited = runtime.image_edit_pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        out_path = resolve_output_path(name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        edited.save(out_path, format="PNG")

        runtime.image_edit_results_store[task_id] = str(out_path)

    except Exception as error:
        runtime.image_edit_results_store[task_id] = f"Error: {error}"