from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io.wavfile import write

from .. import runtime
from ..config import OUTPUT_DIR
from . import context_service
from .text_service import _summarizer_generate_fn


def _build_music_prompt(user_request: str) -> str:
    """Combine the game's design summary with the new request to produce
    a single, self-contained prompt suitable for the music/audio pipeline."""
    context = context_service.load_context()
    summary = context["summary"].strip()

    if not summary:
        return user_request

    distill_prompt = f"""Game design document:
{summary}

New music/sound request: {user_request}

Rewrite the new request as a single, self-contained, descriptive music-generation prompt
(instrumentation, mood, tempo, genre). Incorporate relevant details from the design document
(theme, tone, setting) only if clearly relevant to this specific request. Do not invent
unrelated details. Output ONLY the final prompt text, nothing else.
"""
    return _summarizer_generate_fn(distill_prompt).strip()


def generate_music(task_id: str, prompt: str, name: Optional[str], max_new_tokens: int) -> None:
    try:
        if runtime.music_model is None or runtime.music_processor is None:
            raise RuntimeError("Music model is not loaded yet")

        final_prompt = _build_music_prompt(prompt)
        print(f"Generating music prompt: {final_prompt}")

        inputs = runtime.music_processor(
            text=[final_prompt],
            return_tensors="pt",
        )

        if runtime.device == "cuda":
            inputs = inputs.to(runtime.device)

        audio_values = runtime.music_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        sampling_rate = runtime.music_model.config.audio_encoder.sampling_rate
        audio = audio_values[0, 0].detach().cpu().numpy().astype(np.float32)

        filename = name or f"{task_id}.wav"
        candidate = Path(filename)
        if candidate.suffix == "":
            candidate = Path(f"{candidate.name}.wav")

        out_path = OUTPUT_DIR / candidate.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(out_path), sampling_rate, audio)

        runtime.music_results_store[task_id] = str(out_path)

        context_service.update_context_with_new_turn(
            user_message=prompt,
            assistant_response="sound",
            generate_fn=_summarizer_generate_fn,
        )

    except Exception as error:
        runtime.music_results_store[task_id] = f"Error: {error}"