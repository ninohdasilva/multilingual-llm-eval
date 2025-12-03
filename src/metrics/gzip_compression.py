# Modified: 2025-12-02 - Clarified as text-only metric
"""Gzip compression ratio calculation.

**IMPORTANT: This is a TEXT-ONLY metric, independent of any language model.**

The gzip compression ratio measures the compressibility of UTF-8 encoded source
text using gzip compression (level 6). It characterizes the intrinsic complexity
and redundancy of the text itself, NOT the model's performance.

Compression ratio measures the information density of text. Languages with
higher redundancy compress better (lower ratio), while languages with high
information density compress less (higher ratio).

This metric is independent of the language model and provides a baseline
for understanding the inherent complexity and redundancy of different languages.
It's useful for:
- Understanding language structure
- Comparing information density across languages
- Identifying languages with high morphological complexity
- Providing context for model-dependent metrics

**Key Point:** The gzip ratio will be identical across all models for the same
language, as it only depends on the source text.

Formula: ratio = compressed_size / original_size
"""

import gzip

from tqdm import tqdm


def calculate_gzip_ratio(texts: list[str]) -> float:
    """
    Calculate average gzip compression ratio for a collection of texts.

    The compression ratio indicates how much the text can be compressed.
    Lower ratios indicate more redundancy (easier to compress).
    Higher ratios indicate less redundancy (harder to compress).

    This metric is useful because:
    1. It's model-independent (no LLM needed)
    2. It reveals language structure and redundancy
    3. It correlates with morphological complexity
    4. It provides context for interpreting other metrics

    Args:
        texts: List of text strings to compress

    Returns:
        Average compression ratio (0.0 to 1.0, where lower = more compressible)

    Raises:
        ValueError: If no text provided
    """
    if not texts:
        raise ValueError("No texts provided for compression")

    total_original = 0
    total_compressed = 0

    for text in tqdm(texts, desc="Computing gzip ratio"):
        # Encode as UTF-8 bytes
        text_bytes = text.encode("utf-8")
        original_size = len(text_bytes)

        # Compress with gzip
        compressed = gzip.compress(text_bytes, compresslevel=6)
        compressed_size = len(compressed)

        total_original += original_size
        total_compressed += compressed_size

    if total_original == 0:
        raise ValueError("No text content found")

    # Calculate average ratio
    ratio = total_compressed / total_original

    return ratio
