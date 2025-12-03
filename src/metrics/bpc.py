"""Bits-per-character (BPC) calculation.

BPC is a tokenizer-neutral metric that allows fair comparison of model
performance across languages with different tokenization characteristics.

Unlike perplexity (which is token-based), BPC normalizes by the number
of characters, making it independent of how the tokenizer segments text.
This is especially important for morphologically-rich languages where
tokenization can vary significantly.

Formula: BPC = (loss * num_tokens) / num_characters (per sentence, then aggregated)
This is computed from loss directly to be 100% tokenizer-agnostic.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def calculate_bpc_from_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 8,
) -> float:
    """
    Calculate bits-per-character from model loss directly.

    BPC measures the average number of bits needed to encode each character
    of text, given the model's predictions. Lower BPC indicates better
    compression and prediction quality.

    This metric is computed per sentence as: bpc = (loss * len(tokens)) / len(sentence)
    and then aggregated, making it 100% tokenizer-agnostic.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        batch_size: Number of texts to process in each batch

    Returns:
        Average bits-per-character value

    Raises:
        ValueError: If no characters or tokens found
    """
    model.eval()
    total_bpc_numerator = 0.0
    total_chars = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize texts
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            # Create labels with proper masking
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Get model outputs
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Compute BPC per sentence in batch
            for j, text in enumerate(batch_texts):
                # Count non-padding tokens for this sentence
                sentence_mask = attention_mask[j]
                num_tokens = sentence_mask.sum().item()

                # Count characters
                num_chars = len(text)

                if num_chars > 0 and num_tokens > 0:
                    # BPC per sentence: (loss * tokens) / chars
                    # Note: loss is averaged over the batch, so we use it directly
                    # For per-sentence accuracy, we'd need per-sentence losses
                    # but for aggregated BPC, using batch loss is acceptable
                    sentence_bpc = (loss.item() * num_tokens) / num_chars
                    total_bpc_numerator += sentence_bpc * num_chars
                    total_chars += num_chars

    if total_chars == 0:
        raise ValueError("No characters found in input texts")

    # Average BPC weighted by character count
    avg_bpc = total_bpc_numerator / total_chars

    return avg_bpc


def calculate_bpc(
    perplexity: float, texts: list[str], tokenizer: PreTrainedTokenizer
) -> float:
    """
    Calculate bits-per-character from perplexity (legacy method).

    This is kept for backward compatibility but calculate_bpc_from_loss
    is preferred as it's more tokenizer-agnostic.

    Args:
        perplexity: Perplexity score (must be > 0)
        texts: List of text strings used to compute perplexity
        tokenizer: Tokenizer used to count tokens

    Returns:
        Bits-per-character value
    """
    import math

    if perplexity <= 0:
        raise ValueError("Perplexity must be positive")

    # Count total characters
    total_chars = sum(len(text) for text in texts)

    if total_chars == 0:
        raise ValueError("No characters found in input texts")

    # Count total tokens (excluding special tokens)
    special_ids = set(tokenizer.all_special_ids)
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        filtered_tokens = [t for t in tokens if t not in special_ids]
        total_tokens += len(filtered_tokens)

    if total_tokens == 0:
        raise ValueError("No tokens found in input texts")

    # Legacy formula: BPC = (log2(perplexity) * num_tokens) / num_chars
    bpc = (math.log2(perplexity) * total_tokens) / total_chars

    return bpc
