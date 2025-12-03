#!/bin/bash
# Example execution scripts for multilingual LLM evaluation

# Quick mode: 3 languages, 50 sentences each
echo "Running quick evaluation (3 languages, 50 sentences each)..."
uv run python src/main.py --model HuggingFaceTB/SmolLM3-3B --langs fra,fin,zho_Hans --mode quick --device cuda

# Full mode: 3 languages, 200 sentences each
echo "Running full evaluation (3 languages, 200 sentences each)..."
uv run python src/main.py --model HuggingFaceTB/SmolLM3-3B --langs fra,fin,zho_Hans --mode full --device cuda

