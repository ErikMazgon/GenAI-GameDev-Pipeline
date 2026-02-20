from pathlib import Path
from typing import Optional

from .. import runtime
from ..config import OUTPUT_DIR


def generate_image(task_id: str, prompt: str, name: Optional[str], height: int, width: int) -> None:
    try:
        if runtime.image_pipe is None:
            raise RuntimeError("Image pipeline is not loaded yet")

        image = runtime.image_pipe(prompt, height=height, width=width).images[0]

        filename = name or f"{task_id}.png"
        candidate = Path(filename)
        if candidate.suffix == "":
            candidate = Path(f"{candidate.name}.png")

        out_path = OUTPUT_DIR / candidate.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")

        runtime.image_results_store[task_id] = str(out_path)
    except Exception as error:
        runtime.image_results_store[task_id] = f"Error: {error}"
