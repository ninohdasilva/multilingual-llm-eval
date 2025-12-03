"""HuggingFace transformer baseline evaluation.

Supports GPT-2 and Qwen-1.5B baselines.
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.combined import calculate_perplexity_and_entropy
from metrics.gzip_compression import calculate_gzip_ratio
from metrics.tokenizer_stats import calculate_tokenizer_stats

logger = logging.getLogger(__name__)


class HFTransformerBaseline:
    """HuggingFace transformer baseline evaluator."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        fallback_to_cpu: bool = True,
    ):
        """
        Initialize transformer baseline.

        Args:
            model_name: HuggingFace model identifier (e.g., 'gpt2', 'Qwen/Qwen2-1.5B')
            device: Preferred device ('cuda' or 'cpu')
            fallback_to_cpu: Whether to fallback to CPU if GPU fails
        """
        self.model_name = model_name
        self.device = device
        self.fallback_to_cpu = fallback_to_cpu
        self.model = None
        self.tokenizer = None
        self.actual_device = None

    def load(self) -> None:
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading baseline model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Try to move to preferred device
            try:
                self.model = self.model.to(self.device)
                self.actual_device = self.device
                logger.info(f"Loaded {self.model_name} on {self.device}")
            except Exception as e:
                if self.fallback_to_cpu and self.device != "cpu":
                    logger.warning(
                        f"Failed to load {self.model_name} on {self.device}: {e}. Falling back to CPU."
                    )
                    self.model = self.model.to("cpu")
                    self.actual_device = "cpu"
                else:
                    raise

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            error_msg = f"Failed to load baseline model {self.model_name}: {e}"
            logger.error(error_msg)
            # Check if it's an auth issue for Qwen
            if "qwen" in self.model_name.lower() or "Qwen" in self.model_name:
                logger.warning(
                    f"Qwen model may require authentication. Please check HuggingFace token."
                )
            raise RuntimeError(error_msg)

    def evaluate(
        self, texts: list[str], batch_size: int = 8
    ) -> Tuple[float, float, float, float, dict]:
        """
        Evaluate model on texts.

        Args:
            texts: List of text strings to evaluate
            batch_size: Batch size for evaluation

        Returns:
            Tuple of (perplexity, entropy, bpc, gzip_ratio, tokenizer_stats_dict)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Compute metrics
        ppl, entropy, bpc = calculate_perplexity_and_entropy(
            self.model, self.tokenizer, texts, self.actual_device, batch_size=batch_size
        )

        gzip_ratio = calculate_gzip_ratio(texts)

        tok_stats = calculate_tokenizer_stats(self.tokenizer, texts)

        return ppl, entropy, bpc, gzip_ratio, tok_stats


# Modified: 2025-12-02 - Added tokenization validation
def evaluate_gpt2_baseline(
    texts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
    language_code: Optional[str] = None
) -> Tuple[float, float, float, float, dict, str]:
    """
    Evaluate GPT-2 baseline with tokenization validation.

    Args:
        texts: List of texts to evaluate
        device: Device to use
        batch_size: Batch size
        language_code: FLORES language code for validation (e.g., 'zho_Hans')

    Returns:
        Tuple of (perplexity, entropy, bpc, gzip_ratio, tokenizer_stats_dict, notes)
    """
    baseline = HFTransformerBaseline("gpt2", device=device)
    baseline.load()
    
    notes = ""
    
    # Validate tokenization if language provided
    if language_code and texts:
        from metrics.tokenizer_stats import validate_tokenizer_roundtrip
        success, diagnostic, _, _ = validate_tokenizer_roundtrip(
            baseline.tokenizer, texts[0], language_code
        )
        logger.info(f"GPT-2 tokenization validation for {language_code}: {diagnostic}")
        
        if not success:
            notes = f"gpt2_tokenizer_roundtrip_failed_for_{language_code}"
            logger.warning(
                f"GPT-2 may not support {language_code} properly. "
                f"Metrics may be unreliable. Proceeding with caution."
            )
    
    ppl, entropy, bpc, gzip_ratio, tok_stats = baseline.evaluate(texts, batch_size=batch_size)
    return ppl, entropy, bpc, gzip_ratio, tok_stats, notes


def evaluate_qwen_baseline(
    texts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
    model_name: str = "Qwen/Qwen2-1.5B",
) -> Tuple[float, float, float, float, dict]:
    """
    Evaluate Qwen-1.5B baseline.

    Args:
        texts: List of texts to evaluate
        device: Device to use
        batch_size: Batch size
        model_name: Qwen model identifier (default: Qwen/Qwen2-1.5B)

    Returns:
        Tuple of (perplexity, entropy, bpc, gzip_ratio, tokenizer_stats_dict)
    """
    baseline = HFTransformerBaseline(model_name, device=device, fallback_to_cpu=True)
    try:
        baseline.load()
        return baseline.evaluate(texts, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Qwen baseline failed: {e}")
        # Return None values to indicate failure
        raise

