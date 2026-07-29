from llm_backend import runtime
from llm_backend.services import context_service


def _run_generation(messages: list, max_new_tokens: int = 128) -> str:
    """Shared low-level call to the model — used for both user replies and summarization."""
    inputs = runtime.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(runtime.device)

    outputs = runtime.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.15,
        eos_token_id=runtime.tokenizer.eos_token_id,
        pad_token_id=runtime.tokenizer.pad_token_id or runtime.tokenizer.eos_token_id,
    )
    result = runtime.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return result.strip()


def _summarizer_generate_fn(prompt: str) -> str:
    """Adapter so context_service can call the model with a plain string prompt."""
    messages = [{"role": "user", "content": prompt}]
    return _run_generation(messages, max_new_tokens=200)


def generate_text(task_id: str, prompt: str) -> None:
    try:
        context = context_service.load_context()
        context_str = context_service.build_context_string(context)

        full_user_content = f"{context_str}\n\nUser request: {prompt}" if context_str else prompt

        messages = [
            {
                "role": "system",
                "content": "Answer clearly and briefly in 5 bullet points or less. No extra text."
            },
            {
                "role": "user",
                "content": full_user_content
            }
        ]

        result = _run_generation(messages, max_new_tokens=128)
        runtime.results_store[task_id] = result

        # Fold this exchange into context for next time
        context_service.update_context_with_new_turn(prompt, result, _summarizer_generate_fn)

    except Exception as error:
        runtime.results_store[task_id] = f"Error: {error}"