#!/bin/bash
# Quick evaluation script for multilingual LLM evaluation pipeline
# GitHub: https://github.com/ninohdasilva/multilingual-llm-eval

set -e

echo "=========================================="
echo "Multilingual LLM Evaluation - Quick Test"
echo "=========================================="
echo ""
echo "Running quick 3-language evaluation (fra, fin, zho_Hans) with 50 sentences..."
echo ""

uv run python src/main.py \
    --model Qwen/Qwen2.5-1.5B \
    --langs fra,fin,zho_Hans \
    --mode quick \
    --device cuda

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Reports available in: outputs/report/report_final.html"
echo "Metrics available in: outputs/metrics/metrics_full.csv"
echo ""
