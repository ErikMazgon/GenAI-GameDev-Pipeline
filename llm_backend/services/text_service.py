from .. import runtime


def generate_text(task_id: str, prompt: str) -> None:
    try:
        messages = [
            {
                "role": "system",
                "content": "Answer clearly and briefly in 5 bullet points or less. No extra text."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        inputs = runtime.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(runtime.device)

        outputs = runtime.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=runtime.tokenizer.eos_token_id,
            pad_token_id=runtime.tokenizer.pad_token_id or runtime.tokenizer.eos_token_id,
            # early_stopping=True,
        )
        result = runtime.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        runtime.results_store[task_id] = result.strip()

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        print("Generated tokens:", len(generated))
        print("Last token id:", generated[-1].item())
        print("EOS token id:", runtime.tokenizer.eos_token_id)
        print("Ended with EOS:", generated[-1].item() == runtime.tokenizer.eos_token_id)
    except Exception as error:
        runtime.results_store[task_id] = f"Error: {error}"


# def generate_text_sync(prompt: str) -> str:
#     messages = [{"role": "user", "content": prompt}]
#     inputs = runtime.tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     ).to(runtime.device)

#     outputs = runtime.model.generate(
#         **inputs,
#         max_new_tokens=512,
#         min_new_tokens=32,
#         do_sample=True,
#         temperature=0.8,
#         top_p=0.95,
#         repetition_penalty=1.1,
#         eos_token_id=runtime.tokenizer.eos_token_id,
#         pad_token_id=runtime.tokenizer.pad_token_id or runtime.tokenizer.eos_token_id,
#         early_stopping=True,
#     )
#     result = runtime.tokenizer.decode(
#         outputs[0][inputs["input_ids"].shape[-1] :],
#         skip_special_tokens=True,
#     )
#     return result.strip()
