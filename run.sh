#!/bin/bash
# Example execution scripts for multilingual LLM evaluation pipeline
# GitHub: https://github.com/ninohdasilva/multilingual-llm-eval

set -e

echo "=========================================="
echo "Multilingual LLM Evaluation Pipeline"
echo "=========================================="
echo ""

# Quick mode: 3 languages, 50 sentences each
echo "Running quick evaluation (3 languages, 50 sentences each)..."
uv run python src/main.py \
    --model Qwen/Qwen2.5-1.5B \
    --langs fra,fin,zho_Hans \
    --mode quick \
    --device cuda

echo ""
echo "=========================================="
echo "Quick evaluation complete!"
echo "Reports available in: outputs/report/report_final.html"
echo "=========================================="
echo ""

# Full mode: 3 languages, 200 sentences each
echo "Running full evaluation (3 languages, 200 sentences each)..."
uv run python src/main.py \
    --model Qwen/Qwen2.5-1.5B \
    --langs fra,fin,zho_Hans \
    --mode full \
    --device cuda

echo ""
echo "=========================================="
echo "Full evaluation complete!"
echo "Reports available in: outputs/report/report_final.html"
echo "=========================================="
