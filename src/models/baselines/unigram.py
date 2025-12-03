"""Character unigram baseline model.

Trains a simple unigram character model on the training data and computes
cross-entropy bits per character and perplexity.
"""

import math
from collections import Counter
from typing import List, Tuple


class CharUnigramBaseline:
    """Character unigram baseline model."""

    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize unigram baseline.

        Args:
            epsilon: Smoothing epsilon for unseen characters
        """
        self.epsilon = epsilon
        self.char_counts: Counter = Counter()
        self.total_chars = 0
        self.vocab_size = 0

    def train(self, texts: List[str]) -> None:
        """
        Train the unigram model on concatenated texts.

        Args:
            texts: List of training text strings
        """
        # Concatenate all texts
        all_text = "".join(texts)
        self.char_counts = Counter(all_text)
        self.total_chars = len(all_text)
        self.vocab_size = len(self.char_counts)

    def get_char_probability(self, char: str) -> float:
        """
        Get probability of a character.

        Args:
            char: Character to get probability for

        Returns:
            Probability (with epsilon smoothing for unseen chars)
        """
        count = self.char_counts.get(char, 0)
        # Laplace-like smoothing: (count + epsilon) / (total + epsilon * vocab_size)
        # For unigram, we use simple epsilon smoothing
        prob = (count + self.epsilon) / (self.total_chars + self.epsilon * (self.vocab_size + 1))
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
            for char in text:
                prob = self.get_char_probability(char)
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


def evaluate_unigram_baseline(
    train_texts: List[str], test_texts: List[str]
) -> Tuple[float, float]:
    """
    Evaluate unigram baseline on test texts.

    Args:
        train_texts: Training texts (used to build the model)
        test_texts: Test texts (used for evaluation)

    Returns:
        Tuple of (bpc, perplexity_char)
    """
    model = CharUnigramBaseline()
    model.train(train_texts)
    return model.evaluate(test_texts)

