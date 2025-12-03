#!/usr/bin/env python3
# Modified: 2025-12-02 - GPT-2 Chinese tokenization validation
"""
GPT-2 Chinese Tokenization Sanity Test

Tests whether GPT-2 tokenizer handles Chinese text correctly.
This is a standalone script that can be run independently to diagnose
tokenization issues before running the full evaluation pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer
from datasets import load_dataset
from metrics.tokenizer_stats import validate_tokenizer_roundtrip


def main():
    parser = argparse.ArgumentParser(
        description="Test GPT-2 tokenizer on Chinese text"
    )
    parser.add_argument(
        "--output",
        default="outputs/logs/gpt2_chinese_sanity.log",
        help="Output log file path"
    )
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Loading Chinese sentences from FLORES-200...")
    try:
        flores = load_dataset("facebook/flores", "zho_Hans", split="dev")
        sentences = [str(s).strip() for s in flores["sentence"][:10] if s]
    except Exception as e:
        print(f"ERROR: Failed to load FLORES dataset: {e}")
        sys.exit(1)
    
    print(f"Testing {len(sentences)} sentences...\n")
    
    results = []
    for i, sentence in enumerate(sentences):
        success, diagnostic, tokens, token_ids = validate_tokenizer_roundtrip(
            tokenizer, sentence, "zho_Hans"
        )
        results.append({
            "index": i,
            "success": success,
            "diagnostic": diagnostic,
            "tokens_sample": tokens[:10],
            "token_ids_sample": token_ids[:10],
            "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence
        })
    
    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("GPT-2 Chinese Tokenization Sanity Test\n")
        f.write("=" * 60 + "\n\n")
        
        pass_count = sum(1 for r in results if r["success"])
        f.write(f"Results: {pass_count}/{len(results)} sentences passed\n\n")
        
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            f.write(f"Sentence {r['index']}: {status}\n")
            f.write(f"  Text: {r['sentence']}\n")
            f.write(f"  {r['diagnostic']}\n")
            f.write(f"  Tokens (first 10): {r['tokens_sample']}\n")
            f.write(f"  Token IDs (first 10): {r['token_ids_sample']}\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        if pass_count < len(results) * 0.7:
            f.write("\nCONCLUSION: GPT-2 tokenizer NOT suitable for Chinese.\n")
            f.write("Recommendation: Mark GPT-2 results for zho_Hans as NaN with note.\n")
            f.write("The tokenizer likely encodes Chinese characters as byte-level tokens,\n")
            f.write("which breaks the semantic meaning and makes perplexity unreliable.\n")
        else:
            f.write("\nCONCLUSION: GPT-2 tokenizer acceptable for Chinese.\n")
            f.write("Round-trip validation passed for most sentences.\n")
    
    print(f"Results written to {output_path}")
    print(f"Pass rate: {pass_count}/{len(results)}")
    
    if pass_count < len(results) * 0.7:
        print("\n⚠️  WARNING: GPT-2 tokenizer may not be suitable for Chinese!")
        print("   Check the log file for details.")
        sys.exit(1)
    else:
        print("\n✓ GPT-2 tokenizer validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()

