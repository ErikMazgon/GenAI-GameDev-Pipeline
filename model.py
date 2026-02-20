import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

print("===========================\n")

print("Začetek poganjanja modela...")

#print(f"Python verzija: {sys.version}")

#print(f"CUDA available: {torch.cuda.is_available()}")
""" if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}") """

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use float16 on GPU for efficiency
    device_map="auto",  # Automatically place model on GPU
)

print(f"Model device: {next(model.parameters()).device}")
print("Model naložen ...")

messages = [
    {"role": "user", "content": "tell me a joke"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)  # Move inputs to GPU

outputs = model.generate(    
    **inputs,
    max_new_tokens=512,            # allows long answers
    min_new_tokens=32,             # forces at least some length
    do_sample=True,                # enables probabilistic generation
    temperature=0.8,               # slightly creative but stable
    top_p=0.95,                    # nucleus sampling
    repetition_penalty=1.1,        # avoids loops
    eos_token_id=tokenizer.eos_token_id,  # ensures correct stop token
    pad_token_id=tokenizer.pad_token_id,  # prevents warnings
    early_stopping=True,           # stop only when model predicts EOS
)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

print(result)

# with open("odgovor.txt", "w", encoding="utf-8") as file:
#     file.write(result)