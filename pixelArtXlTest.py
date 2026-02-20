import os

# set HF cache before importing torch/diffusers
os.environ["HF_HOME"] = r"D:\ErikM\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\ErikM\hf_cache\transformers"
os.environ["XDG_CACHE_HOME"] = r"D:\ErikM\hf_cache"

import torch
from diffusers import DiffusionPipeline
from pathlib import Path  # added

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.bfloat16, device_map="cuda")
pipe.load_lora_weights("nerijs/pixel-art-xl")

prompt = "pixel art, tree, white background, 8-bit"
name = "treeNormal.png"
image = pipe(prompt).images[0]
# image = pipe(prompt, height=512, width=512).images[0]

# save to the specified outputs folder
out_dir = Path(r"C:\Users\erik\Documents\Unreal Projects\MyProject\Content\Python\outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / name
image.save(out_path)