import os
from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_ROOT = Path(r"D:\ErikM\hf_cache")
OUTPUT_DIR = Path(r"C:\Users\erik\Documents\Unreal Projects\EvalGameAI\Plugins\GenAIAssistant\Content\Python\outputs")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MUSIC_MODEL_NAME = "facebook/musicgen-small"
HOST = "127.0.0.1"
PORT = 5000


def configure_cache_paths() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_ROOT / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(HF_CACHE_ROOT)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
