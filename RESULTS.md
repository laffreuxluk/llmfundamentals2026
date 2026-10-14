# Session 1 – Results

Branch: `lucas-noury-session1` · Model for scripts 1–4: `anthropic/claude-haiku-4.5` (via OpenRouter)

All five scripts were run with a real API key on 2026-10-14.

## 1. API connection
`1_check_api_connection_openrouter_api_key.py` returns "Bonjour!", the connection works.

## 2. Temperature and top-p
`2_changing_temperature.py`: 3 runs for each value (temperature 0, 0.5, 1.0 and top_p 0.1, 0.5, 1.0).
At low values the answers are nearly identical from one run to the next; at `top_p=1.0` the chosen cartoon character changes every run (Mufasa, Leslie Knope, Uncle Iroh). Higher randomness gives more variety, which suits brainstorming but not extraction or classification.

## 3. Cost estimation
`3_estimating_costs.py` (prices per 1M tokens: Haiku 4.5 = $1 input / $5 output, Sonnet 5.5 = $2 / $10, from the Anthropic pricing page).

| Item | Value |
|---|---|
| Estimated input (~4 chars/token) | ~6 tokens, $0.000006 |
| Worst-case output (`max_tokens=1000`) | $0.005 |
| Actual call | 12 in + 8 out tokens, $0.000052 |

## 4. Prompting patterns
`4_prompting_patterns.py`: each strategy on the three datasets in `docs/session1`, scored automatically (exact label, final `ANSWER:` value, and the 5 style rules).

| Task | Strategy | Correct | Tokens in / out |
|---|---|---|---|
| Classification (12) | zero-shot | 9/12 | 665 / 48 |
| | few-shot | 10/12 | 1553 / 48 |
| | **system prompt (rules)** | **12/12** | 1481 / 48 |
| Arithmetic (8) | zero-shot | 0/8 | 519 / 160 |
| | few-shot | 2/8 | 999 / 92 |
| | **chain-of-thought** | **8/8** | 599 / 2207 |
| Style (6) | zero-shot | 0/6 | 316 / 360 |
| | few-shot | 2/6 | 1024 / 158 |
| | **system prompt (rules)** | **3/6** | 616 / 128 |

Takeaways
- Classification with rules: putting the rules in the system prompt beats examples alone.
- Multi-step arithmetic needs reasoning: chain-of-thought is the only strategy that works, but it costs about 20x more output tokens than few-shot.
- Style: zero-shot cannot know the café's rules. Few-shot copies the tone but often misses the signature or the length. The system prompt with explicit rules works best, though not perfectly.

## 5. Comparing LLMs
`5_comparing_llms.py` → `src/llmfundamentals/comparing_llms.csv` (3 prompts × 4 models).

| Model | Avg cost | Avg latency |
|---|---|---|
| claude-haiku-4.5 | $0.000589 | 2.01 s |
| claude-sonnet-5.5 | $0.001616 | 2.36 s |
| gpt-4.1-mini | $0.000145 | 1.52 s |
| gpt-4.1-nano | $0.000037 | 1.60 s |

Cheapest: `gpt-4.1-nano`. Fastest: `gpt-4.1-mini`. On these three simple prompts the nano answers are correct (balance sheet definition, email, 120 × 65% = 78 on site), so the cheapest model is good enough here. Sonnet is the most expensive without a visible quality gain on such easy tasks.

## Notes
- OpenAI prices in `src/common/llmwrapper.py` (gpt-4.1-mini $0.40/$1.60, gpt-4.1-nano $0.10/$0.40) were not checked against the OpenRouter page, so the costs in section 5 depend on them.
- Style answers are capped at 60 tokens, which can truncate long zero-shot replies.
