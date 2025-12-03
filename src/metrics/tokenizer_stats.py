# Modified: 2025-12-02 - Added tokenizer round-trip validation
"""Tokenizer statistics calculation.

Tokenizer statistics reveal how the tokenizer segments text differently
across languages. This is crucial for understanding:

1. Tokenizer bias: Some languages require more tokens per character
2. Morphological complexity: Agglutinative languages often tokenize
   into more pieces than isolating languages
3. Cross-lingual fairness: Token-based metrics (like perplexity) can
   be biased by tokenization differences

Key metrics:
- avg_tokens: Average tokens per sentence
- tokens_per_char: Ratio showing tokenization efficiency
- avg_char_length: Average sentence length in characters
"""

import logging
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def calculate_tokenizer_stats(
    tokenizer: PreTrainedTokenizer, texts: list[str]
) -> dict[str, float]:
    """
    Calculate tokenizer statistics for a collection of texts.

    These statistics help understand:
    - How efficiently the tokenizer segments each language
    - Whether tokenization bias affects metric comparisons
    - The relationship between character and token counts

    For morphologically-rich languages (e.g., Turkish, Finnish),
    tokenization often produces more tokens per character because:
    - Words are longer (more morphemes)
    - Subword tokenization breaks complex words into many pieces
    - The tokenizer may not have seen all morphological variants

    Args:
        tokenizer: Tokenizer to analyze
        texts: List of text strings to tokenize

    Returns:
        Dictionary with:
        - avg_tokens: Average number of tokens per sentence
        - tokens_per_char: Average ratio of tokens to characters
        - avg_char_length: Average sentence length in characters
    """
    if not texts:
        raise ValueError("No texts provided")

    total_tokens = 0
    total_chars = 0
    num_sentences = len(texts)

    for text in tqdm(texts, desc="Computing tokenizer stats"):
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Remove special tokens (BOS, EOS, PAD, etc.) from token count
        special_ids = set(tokenizer.all_special_ids)
        filtered_tokens = [t for t in tokens if t not in special_ids]
        num_tokens = len(filtered_tokens)

        # Count characters
        num_chars = len(text)

        total_tokens += num_tokens
        total_chars += num_chars

    if num_sentences == 0:
        raise ValueError("No sentences found")

    avg_tokens = total_tokens / num_sentences
    avg_char_length = total_chars / num_sentences
    tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0.0

    return {
        "avg_tokens": avg_tokens,
        "tokens_per_char": tokens_per_char,
        "avg_char_length": avg_char_length,
    }


def validate_tokenizer_roundtrip(
    tokenizer: PreTrainedTokenizer, text: str, language_code: str
) -> tuple[bool, str, list[str], list[int]]:
    """
    Validate tokenizer round-trip encoding/decoding.

    Tests whether a tokenizer can properly encode and decode text,
    which is critical for ensuring metrics are meaningful.

    Args:
        tokenizer: Tokenizer to validate
        text: Sample text to test
        language_code: FLORES language code (e.g., 'zho_Hans', 'fra')

    Returns:
        - success (bool): Whether round-trip is valid
        - diagnostic (str): Diagnostic message
        - tokens (list[str]): Token strings
        - token_ids (list[int]): Token IDs
    """
    try:
        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

        # For CJK languages, check character preservation
        if language_code in ["zho_Hans", "zho_Hant", "jpn", "kor"]:
            # Count overlapping characters
            original_chars = set(text)
            decoded_chars = set(decoded)
            overlap = (
                len(original_chars & decoded_chars) / len(original_chars)
                if original_chars
                else 0
            )
            success = overlap >= 0.7 and len(decoded) > 0
            diagnostic = f"CJK round-trip: {overlap:.1%} char overlap, decoded_len={len(decoded)}, original_len={len(text)}"
        else:
            # For non-CJK, check word-level preservation
            original_words = text.split()[:3]  # Check first 3 words
            success = len(decoded) > 0 and any(
                word in decoded for word in original_words if word
            )
            diagnostic = (
                f"Round-trip: decoded_len={len(decoded)}, original_len={len(text)}"
            )

        return success, diagnostic, tokens, token_ids

    except Exception as e:
        logger.error(f"Round-trip validation failed with exception: {e}")
        return False, f"Exception: {str(e)}", [], []
