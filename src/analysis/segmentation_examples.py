# Modified: 2025-12-02 - Added JSONL export for segmentation examples
"""Generate per-language segmentation examples for the report."""

import json
from pathlib import Path
from transformers import PreTrainedTokenizer


def generate_segmentation_examples(
    tokenizer: PreTrainedTokenizer, sentences: list[str], max_examples: int = 2
) -> list[dict]:
    """
    Generate segmentation examples for a language.

    Args:
        tokenizer: Tokenizer to use
        sentences: List of sentences to tokenize
        max_examples: Maximum number of examples to return

    Returns:
        List of dictionaries with:
        - original: original sentence
        - tokens: list of token strings
        - num_tokens: number of tokens
        - tokens_per_char: tokens per character for this sentence
    """
    examples = []
    special_ids = set(tokenizer.all_special_ids)

    for sentence in sentences[:max_examples]:
        # Tokenize
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        # Filter special tokens
        filtered_token_ids = [tid for tid in token_ids if tid not in special_ids]

        # Decode tokens to strings
        token_strings = tokenizer.convert_ids_to_tokens(filtered_token_ids)

        # Compute tokens per char for this sentence
        num_chars = len(sentence)
        tokens_per_char = len(filtered_token_ids) / num_chars if num_chars > 0 else 0.0

        examples.append(
            {
                "original": sentence,
                "tokens": token_strings,
                "num_tokens": len(filtered_token_ids),
                "tokens_per_char": tokens_per_char,
            }
        )

    return examples


def save_segmentation_examples_jsonl(
    language_code: str,
    model_name: str,
    sentences: list[str],
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    max_examples: int = 2,
) -> Path:
    """
    Generate and save segmentation examples to JSONL.

    Creates a JSONL file with tokenization examples showing how the
    tokenizer segments text. Each line is a JSON object with sentence
    details, tokens, and statistics.

    Args:
        language_code: FLORES language code (e.g., 'fra', 'zho_Hans')
        model_name: Model identifier (e.g., 'gpt2', 'HuggingFaceTB/SmolLM3-3B')
        sentences: List of sentences to tokenize
        tokenizer: Tokenizer to use
        output_dir: Directory to save JSONL file
        max_examples: Maximum number of examples to save (default: 2)

    Returns:
        Path to saved JSONL file
    """
    output_path = output_dir / f"segmentation_examples_{language_code}.jsonl"
    special_ids = set(tokenizer.all_special_ids)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sentence in enumerate(sentences[:max_examples]):
            # Tokenize
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            filtered_token_ids = [tid for tid in token_ids if tid not in special_ids]
            token_strings = tokenizer.convert_ids_to_tokens(filtered_token_ids)

            # Calculate statistics
            num_chars = len(sentence)
            tokens_per_char = (
                len(filtered_token_ids) / num_chars if num_chars > 0 else 0.0
            )

            # Create record
            record = {
                "language": language_code,
                "sentence_index": idx,
                "model": model_name,
                "original": sentence,
                "tokens": " | ".join(token_strings),  # Pipe-separated for readability
                "token_ids": ",".join(map(str, filtered_token_ids)),
                "num_tokens": len(filtered_token_ids),
                "num_chars": num_chars,
                "tokens_per_char": round(tokens_per_char, 4),
            }

            # Write as JSON line
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    return output_path
