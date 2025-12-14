# Evaluation

Evaluate image editing quality using VIEScore and pixel-level metrics.

## Quick Start

```bash
cd src/evaluation

python evaluation_metrics.py \
    --model_json_path <path_to_test_json> \
    --backbone gpt4o
```

## Metrics

| Metric | Description |
|--------|-------------|
| SC | Instruction satisfaction (0-10) |
| PQ | Content consistency (0-10) |
| Overall | √(SC × PQ) (0-10) |
| L1/L2 | Pixel-level loss |

## Configuration

- **GPT-4o**: Set `--api_key` or use OpenAI API Key environment variable
- **Google Gemini**: Set `GOOGLE_API_KEY` environment variable
- **Qwen2.5-VL**: Requires DashScope API Key via `--api_key`