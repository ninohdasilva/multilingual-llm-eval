# Modified: 2025-12-02 - Added --report-only flag for quick iteration
"""Command-line interface argument parsing."""

import argparse
from typing import Optional


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments according to specification.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual LLM on FLORES-200 dataset with baselines"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="HuggingFace model identifier (e.g., Qwen/Qwen2.5-1.5B). Required unless --report-only is used.",
    )

    parser.add_argument(
        "--langs",
        type=str,
        required=False,
        help="Comma-separated list of FLORES language codes (e.g., fra,fin,zho_Hans). Required unless --report-only is used.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick"],
        default="full",
        help="Evaluation mode: 'full' (200 sentences) or 'quick' (50 sentences). Default: full",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Alias for --mode quick (overrides --mode if specified)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use: 'cuda' or 'cpu'. If not specified, auto-detect (cuda if available, else cpu)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip evaluation and only regenerate plots/reports from existing metrics_full.csv. Useful for quick iteration on visualizations.",
    )

    args = parser.parse_args()

    # Validate required arguments unless in report-only mode
    if not args.report_only:
        if not args.model:
            parser.error("--model is required (unless using --report-only)")
        if not args.langs:
            parser.error("--langs is required (unless using --report-only)")

    # Handle --quick flag (overrides --mode)
    if args.quick:
        args.mode = "quick"

    # Determine device if not specified
    if args.device is None:
        from config import detect_device

        args.device = detect_device()

    return args

