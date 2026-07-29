import json
import os

CONTEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context")
SUMMARY_PATH = os.path.join(CONTEXT_DIR, "summary.txt")
RECENT_TURNS_PATH = os.path.join(CONTEXT_DIR, "recent_turns.json")

MAX_VERBATIM_TURNS = 4
MAX_SUMMARY_CHARS = 4000  # soft guard so the structured doc doesn't grow unbounded

os.makedirs(CONTEXT_DIR, exist_ok=True)

GAME_MODEL_TEMPLATE = """## Game Overview
- Title:
- Genre:
- Theme/Tone:

## World & Setting
-

## Characters
-

## Mechanics & Gameplay
-

## Art & Audio Style
-

## Assets Generated
- Images:
- Sounds:

## Other Notes
-
"""


def load_context() -> dict:
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = f.read()
    else:
        summary = GAME_MODEL_TEMPLATE

    if os.path.exists(RECENT_TURNS_PATH):
        with open(RECENT_TURNS_PATH, "r", encoding="utf-8") as f:
            raw_recent_turns = f.read().strip()

        if raw_recent_turns:
            try:
                recent_turns = json.loads(raw_recent_turns)
            except json.JSONDecodeError:
                recent_turns = []
        else:
            recent_turns = []
    else:
        recent_turns = []

    return {"summary": summary, "recent_turns": recent_turns}


def save_context(context: dict) -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(context["summary"])
    with open(RECENT_TURNS_PATH, "w", encoding="utf-8") as f:
        json.dump(context["recent_turns"], f, indent=2)


def build_context_string(context: dict) -> str:
    parts = [f"Game design document so far:\n{context['summary'].strip()}"]
    if context["recent_turns"]:
        turn_lines = [
            f"User: {t['user']}\nResponse: {t['assistant']}"
            for t in context["recent_turns"]
        ]
        parts.append("Recent exchanges:\n" + "\n\n".join(turn_lines))
    return "\n\n".join(parts)


def _update_game_model(old_summary: str, new_turn: dict, generate_fn) -> str:
    prompt = f"""You maintain a structured design document describing a video game the user is building,
based on their requests so far (text descriptions, image generations, and sound generations).

Current design document:
{old_summary.strip()}

New request just made:
User: {new_turn['user']}
Response type: {new_turn['assistant']}

Update the design document. Rules:
- Keep the exact same section headers, in the same order. Do not add new sections or remove existing ones.
- Only add or revise facts that are clearly stated or strongly implied by the new request.
- Never delete existing facts unless the new request clearly contradicts them.
- Do not invent details that weren't stated.
- Output ONLY the full updated document, same section format, nothing else.
"""
    updated = generate_fn(prompt).strip()
    return updated[:MAX_SUMMARY_CHARS]


def update_context_with_new_turn(user_message: str, assistant_response: str, generate_fn) -> None:
    context = load_context()
    new_turn = {"user": user_message, "assistant": assistant_response}

    # Update the persistent game model every single turn
    context["summary"] = _update_game_model(context["summary"], new_turn, generate_fn)

    # Keep a short rolling window of raw recent turns
    context["recent_turns"].append(new_turn)
    if len(context["recent_turns"]) > MAX_VERBATIM_TURNS:
        context["recent_turns"].pop(0)

    save_context(context)