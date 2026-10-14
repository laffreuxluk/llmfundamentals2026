import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 100
# PROMPT = "Describe winter in one poetic sentence.""
PROMPT = "What cartoon character would make the best president?"
TEMPERATURES = [0.0, 0.5, 1.0]
TOP_PS = [0.1, 0.5, 1.0]

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def ask(prompt, **sampling):
    """sampling is either temperature=... or top_p=... (not both)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        extra_body=sampling,
    )
    return response.content[0].text


def main():
    # Temperature: 3 runs per value, to see how much the answer varies
    for temperature in TEMPERATURES:
        for run in range(3):
            logger.info(
                "temperature={} run {}: {}", temperature, run + 1, ask(PROMPT, temperature=temperature)
            )

    # Top p: 3 runs per value
    for top_p in TOP_PS:
        for run in range(3):
            logger.info("top_p={} run {}: {}", top_p, run + 1, ask(PROMPT, top_p=top_p))


if __name__ == "__main__":
    main()
