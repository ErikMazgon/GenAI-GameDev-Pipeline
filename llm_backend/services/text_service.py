from .. import runtime


def generate_text(task_id: str, prompt: str) -> None:
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = runtime.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(runtime.device)

        outputs = runtime.model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=runtime.tokenizer.eos_token_id,
            pad_token_id=runtime.tokenizer.pad_token_id,
            early_stopping=True,
        )
        result = runtime.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
        runtime.results_store[task_id] = result.strip()
    except Exception as error:
        runtime.results_store[task_id] = f"Error: {error}"


def generate_text_sync(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = runtime.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(runtime.device)

    outputs = runtime.model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=32,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=runtime.tokenizer.eos_token_id,
        pad_token_id=runtime.tokenizer.pad_token_id,
        early_stopping=True,
    )
    result = runtime.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    return result.strip()
