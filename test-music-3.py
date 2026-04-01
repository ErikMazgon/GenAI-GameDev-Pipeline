import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write

model_name = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name)

inputs = processor(
    text=["magic forest for robin hood game 8-bit"],
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
audio = audio_values[0, 0].cpu().numpy()

write("outputs/output.wav", sampling_rate, audio)