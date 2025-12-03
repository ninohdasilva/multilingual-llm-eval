#!/bin/bash
# Evaluation script for multilingual LLM evaluation pipeline

set -e

echo "=========================================="
echo "Multilingual LLM Evaluation - Quick Test"
echo "=========================================="
echo ""
echo "Running quick 3-language evaluation (fin, fra, zho) with 50 sentences..."
uv run python src/main.py --model "HuggingFaceTB/SmolLM3-3B" --langs fin,fra,zho --quick

echo ""
echo "=========================================="
echo "Multilingual LLM Evaluation - Full Run"
echo "=========================================="
echo ""
echo "Running full 7-language evaluation with 200 sentences per language..."
uv run python src/main.py --model "HuggingFaceTB/SmolLM3-3B" --full

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Reports available in: outputs/report.html"

