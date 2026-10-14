import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MAX_TOKENS = 1000
PROMPT = "Say hello in French."

# Pricing page: https://platform.claude.com/docs/en/about-claude/pricing
# Prices in USD per 1M tokens
PRICES = {
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},
}
CHARS_PER_TOKEN = 4  # rule of thumb for English text

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)
messages = [{"role": "user", "content": PROMPT}]


def cost_usd(model, input_tokens, output_tokens):
    price = PRICES[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def main():
    model = "anthropic/claude-haiku-4.5"

    # Before the call: estimate the input tokens, and the worst case for the output
    estimated_input_tokens = len(PROMPT) // CHARS_PER_TOKEN + 1
    logger.info(
        "Estimated input cost: ~{} tokens -> ${:.8f}",
        estimated_input_tokens,
        cost_usd(model, estimated_input_tokens, 0),
    )
    logger.info(
        "Max output cost with max_tokens={}: ${:.8f}",
        MAX_TOKENS,
        cost_usd(model, 0, MAX_TOKENS),
    )

    # Actual cost
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, messages=messages
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    logger.info(
        "Actual: {} input + {} output tokens -> ${:.8f}",
        input_tokens,
        output_tokens,
        cost_usd(model, input_tokens, output_tokens),
    )


if __name__ == "__main__":
    main()
