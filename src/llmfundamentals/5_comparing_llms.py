"""
Compare LLMs from Anthropic and OpenAI on the same prompts, using your LLMWrapper.
"""
import csv
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.common.llmwrapper import LLMWrapper

# (provider, model)
MODELS = [
    ("anthropic", "anthropic/claude-haiku-4.5"),
    ("anthropic", "anthropic/claude-sonnet-5.5"),
    ("openai", "openai/gpt-4.1-mini"),
    ("openai", "openai/gpt-4.1-nano"),
]
PROMPTS = [
    "Explain in two sentences what a balance sheet is.",
    "Write a short professional email asking a professor for an extension on a report.",
    "A company has 120 employees and 35% work remotely. How many work on site?",
]
OUTPUT_FILE = Path(__file__).parent / "comparing_llms.csv"

load_dotenv()  # reads the .env file


def main():
    rows = []
    for provider, model in MODELS:
        llm = LLMWrapper(provider, model=model)
        for prompt in PROMPTS:
            response = llm.ask(prompt)
            logger.info("{} | {!r}: {}", model, prompt, response.text)
            rows.append(
                [
                    model,
                    prompt,
                    response.text.replace("\n", " "),
                    response.input_tokens,
                    response.output_tokens,
                    response.cost_usd,
                    round(response.latency_seconds, 2),
                ]
            )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["model", "prompt", "answer", "input_tokens", "output_tokens", "cost_usd", "latency_s"]
        )
        writer.writerows(rows)
    logger.info("Saved {} rows to {}", len(rows), OUTPUT_FILE)

    # One line per model: average cost and average latency
    averages = {}
    for _, model in MODELS:
        model_rows = [r for r in rows if r[0] == model]
        avg_cost = sum(r[5] or 0 for r in model_rows) / len(model_rows)
        avg_latency = sum(r[6] for r in model_rows) / len(model_rows)
        averages[model] = (avg_cost, avg_latency)
        logger.info("{} | avg cost ${:.6f} | avg latency {:.2f}s", model, avg_cost, avg_latency)

    cheapest = min(averages, key=lambda m: averages[m][0])
    fastest = min(averages, key=lambda m: averages[m][1])
    logger.info("Cheapest: {} | Fastest: {}", cheapest, fastest)
    # Is the answer of the cheapest one good enough? Read the CSV and judge it yourself.


if __name__ == "__main__":
    main()
