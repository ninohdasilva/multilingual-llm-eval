"""Character n-gram baseline model (5-gram with Laplace smoothing).

Implements a character-level n-gram model for baseline comparison.
Uses n=5 with Laplace smoothing (alpha=1).
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple


class CharNgramBaseline:
    """Character n-gram baseline model."""

    def __init__(self, n: int = 5, alpha: float = 1.0):
        """
        Initialize n-gram baseline.

        Args:
            n: N-gram order (default: 5)
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.n = n
        self.alpha = alpha
        self.ngram_counts: Dict[str, int] = defaultdict(int)
        self.context_counts: Dict[str, int] = defaultdict(int)
        self.vocab_size = 0

    def train(self, texts: List[str]) -> None:
        """
        Train the n-gram model on concatenated texts.

        Args:
            texts: List of training text strings
        """
        # Concatenate all texts
        all_text = "".join(texts)

        # Build character vocabulary
        chars = set(all_text)
        self.vocab_size = len(chars)

        # Count n-grams and contexts
        for i in range(len(all_text) - self.n + 1):
            ngram = all_text[i : i + self.n]
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def get_ngram_probability(self, ngram: str) -> float:
        """
        Get probability of an n-gram using Laplace smoothing.

        Args:
            ngram: N-gram string (length n)

        Returns:
            Probability with Laplace smoothing
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram must have length {self.n}, got {len(ngram)}")

        context = ngram[:-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        # Laplace smoothing: P(char|context) = (count(ngram) + alpha) / (count(context) + alpha * vocab_size)
        prob = (ngram_count + self.alpha) / (
            context_count + self.alpha * self.vocab_size
        )
        return prob

    def compute_cross_entropy_bits_per_char(self, texts: List[str]) -> float:
        """
        Compute cross-entropy in bits per character.

        Args:
            texts: List of test texts

        Returns:
            Average cross-entropy in bits per character
        """
        total_bits = 0.0
        total_chars = 0

        for text in texts:
            # Process text with n-gram sliding window
            for i in range(self.n - 1, len(text)):
                ngram = text[i - self.n + 1 : i + 1]
                prob = self.get_ngram_probability(ngram)
                # Cross-entropy: -log2(p)
                bits = -math.log2(prob) if prob > 0 else float("inf")
                total_bits += bits
                total_chars += 1

        if total_chars == 0:
            return 0.0

        return total_bits / total_chars

    def compute_perplexity_char(self, texts: List[str]) -> float:
        """
        Compute character-level perplexity.

        Args:
            texts: List of test texts

        Returns:
            Character-level perplexity
        """
        # Perplexity = 2^(cross_entropy_bits)
        ce_bits = self.compute_cross_entropy_bits_per_char(texts)
        return 2.0 ** ce_bits

    def evaluate(self, texts: List[str]) -> Tuple[float, float]:
        """
        Evaluate on test texts and return BPC and perplexity.

        Args:
            texts: List of test texts

        Returns:
            Tuple of (bpc, perplexity_char)
        """
        bpc = self.compute_cross_entropy_bits_per_char(texts)
        ppl = self.compute_perplexity_char(texts)
        return bpc, ppl


def evaluate_char_ngram_baseline(
    train_texts: List[str], test_texts: List[str], n: int = 5
) -> Tuple[float, float]:
    """
    Evaluate char n-gram baseline on test texts.

    Args:
        train_texts: Training texts (used to build the model)
        test_texts: Test texts (used for evaluation)
        n: N-gram order (default: 5)

    Returns:
        Tuple of (bpc, perplexity_char)
    """
    model = CharNgramBaseline(n=n, alpha=1.0)
    model.train(train_texts)
    return model.evaluate(test_texts)

