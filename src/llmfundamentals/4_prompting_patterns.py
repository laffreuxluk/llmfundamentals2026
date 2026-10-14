import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def ask(prompt, system=None, max_tokens=300):
    """Send one prompt. Returns (reply text, input tokens, output tokens)."""
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


# Zero-shot: just the instruction, no worked example.
ZERO_SHOT = """Classify the message as: complaint, question or praise.

Message: Thanks, the support team was great!
Category:"""

# Few-shot: the same task with several worked examples.
FEW_SHOT = """Classify the message as: complaint, question or praise.

Message: My parcel arrived broken.
Category: complaint

Message: Where can I find my invoice?
Category: question

Message: I love the new app design.
Category: praise

Message: Thanks, the support team was great!
Category:"""

# Role/system: a system prompt sets the role and the rules; the user turn is only the input.
SYSTEM = """You are a customer-support triage assistant.
Classify each message as exactly one of: complaint, question, praise.
Reply with the category only, in lowercase."""
ROLE_PROMPT = "Message: Thanks, the support team was great!"

# Chain of thought: ask the model to reason step by step before giving the answer.
CHAIN_OF_THOUGHT = """A train leaves at 14:45 and the trip takes 2 hours 50 minutes.
A taxi from the station to the hotel takes 25 minutes.
At what time do I reach the hotel?

Think step by step, then end with a line of the form 'Final answer: <time>'."""

# name -> (system prompt, user prompt, max_tokens)
PROMPTS = {
    "zero-shot": (None, ZERO_SHOT, 10),
    "few-shot": (None, FEW_SHOT, 10),
    "role/system": (SYSTEM, ROLE_PROMPT, 10),
    "chain of thought": (None, CHAIN_OF_THOUGHT, 300),
}


DATA = Path(__file__).resolve().parents[2] / "docs" / "session1"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


CLS = load("classification")
ARI = load("arithmetic")
STY = load("style")


# --- Task prompts: one builder per (task, strategy) -> (system, user, max_tokens) ---

SIGNATURE = " ~ hc"


def classification_prompt(strategy, text):
    instruction = CLS["instruction"]
    if strategy == "zero-shot":
        return None, f"{instruction}\n\nMessage: {text}\nCategory:", 10
    if strategy == "few-shot":
        shots = "\n\n".join(
            f"Message: {e['input']}\nCategory: {e['label']}" for e in CLS["examples"]
        )
        return None, f"{instruction}\n\n{shots}\n\nMessage: {text}\nCategory:", 10
    # system: the rules go in the system prompt, the user turn is only the input
    system = f"{instruction}\n\nRules: {CLS['rules']}"
    return system, f"Message: {text}", 10


def arithmetic_prompt(strategy, question):
    answer_format = "End with a last line of the form 'ANSWER: <number>'."
    if strategy == "zero-shot":
        return None, f"{question}\n\nReply with 'ANSWER: <number>' only.", 20
    if strategy == "few-shot":
        ex = ARI["examples"][0]
        shot = f"Question: {ex['question']}\nANSWER: {ex['answer']}"
        return None, f"{shot}\n\nQuestion: {question}\nReply with 'ANSWER: <number>' only.", 20
    # chain of thought: reason first, then answer
    return None, f"{question}\n\nThink step by step. {answer_format}", 600


def style_prompt(strategy, text):
    instruction = STY["instruction"]
    if strategy == "zero-shot":
        return None, f"{instruction}\n\nNotice: {text}\nRewritten:", 60
    if strategy == "few-shot":
        shots = "\n\n".join(
            f"Notice: {e['input']}\nRewritten: {e['output']}" for e in STY["examples"]
        )
        return None, f"{instruction}\n\n{shots}\n\nNotice: {text}\nRewritten:", 60
    system = f"{instruction}\n\nRules: {STY['rules']}\nReply with the rewritten notice only."
    return system, f"Notice: {text}", 60


# --- Scoring ---


def score_classification(reply, item):
    return reply.strip().lower().rstrip(".") == item["label"]


def score_arithmetic(reply, item):
    lines = [ln for ln in reply.strip().splitlines() if "ANSWER" in ln.upper()]
    if not lines:
        return False
    given = lines[-1].upper().split("ANSWER", 1)[1].lstrip(" :*").strip().rstrip(".")
    expected = item["answer"].strip()
    try:  # numeric answers: compare as numbers
        return float(given.split()[0].replace(",", ".")) == float(expected)
    except (ValueError, IndexError):  # other formats, e.g. a time like 01:55
        return given.replace(" ", "") == expected.upper().replace(" ", "")


def score_style(reply, item):
    """Checks the five rules of the chalkboard style."""
    text = reply.strip()
    if not text.endswith(SIGNATURE):
        return False
    body = text[: -len(SIGNATURE)]
    return (
        text == text.lower()
        and "\n" not in text
        and len(body.split()) <= 12
        and not body.rstrip().endswith(".")
    )


TASKS = {
    # name -> (prompt builder, item field holding the input, scorer, strategies)
    "classification": (
        classification_prompt, "input", score_classification, ["zero-shot", "few-shot", "system"]
    ),
    "arithmetic": (
        arithmetic_prompt, "question", score_arithmetic, ["zero-shot", "few-shot", "chain-of-thought"]
    ),
    "style": (style_prompt, "input", score_style, ["zero-shot", "few-shot", "system"]),
}
DATASETS = {"classification": CLS, "arithmetic": ARI, "style": STY}


def main():
    # EXAMPLE
    for name, (system, prompt, max_tokens) in PROMPTS.items():
        text, tokens_in, tokens_out = ask(prompt, system, max_tokens)
        logger.info(
            "=== {} ({} tokens in, {} tokens out) ===\n{}",
            name,
            tokens_in,
            tokens_out,
            text,
        )

    # Compare the prompting strategies on the three tasks
    summary = []
    for task, (build, field, scorer, strategies) in TASKS.items():
        items = DATASETS[task]["items"]
        for strategy in strategies:
            correct = total_in = total_out = 0
            for item in items:
                system, prompt, max_tokens = build(strategy, item[field])
                reply, tokens_in, tokens_out = ask(prompt, system, max_tokens)
                ok = scorer(reply, item)
                correct += ok
                total_in += tokens_in
                total_out += tokens_out
                logger.debug("{} | {} | {} | {!r}", task, strategy, "OK" if ok else "KO", reply)
            summary.append((task, strategy, correct, len(items), total_in, total_out))
            logger.info(
                "{:<14} {:<17} {}/{} correct | {} tokens in, {} tokens out",
                task, strategy, correct, len(items), total_in, total_out,
            )

    # Which strategy wins on each task? What does it cost in tokens?
    for task in TASKS:
        rows = [r for r in summary if r[0] == task]
        best = max(rows, key=lambda r: (r[2], -(r[4] + r[5])))
        logger.info("Best for {}: {} ({}/{})", task, best[1], best[2], best[3])


if __name__ == "__main__":
    main()
