"""Compute per-sentence metrics for detailed analysis and visualization."""

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_per_sentence_losses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str,
) -> list[float]:
    """
    Compute loss for each sentence individually.

    This is used for loss distribution plots to show variance within a language.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate
        device: Device to run computation on

    Returns:
        List of loss values, one per sentence
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing per-sentence losses"):
            # Tokenize single sentence
            encodings = tokenizer(
                text,
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

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            losses.append(loss.item())

    return losses


def compute_token_frequencies(
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    top_k: int = 50,
) -> dict[int, int]:
    """
    Compute token frequency distribution across texts.

    Args:
        tokenizer: Tokenizer to use
        texts: List of text strings
        top_k: Number of top tokens to return

    Returns:
        Dictionary mapping token_id -> frequency count
    """
    token_counts: dict[int, int] = {}
    special_ids = set(tokenizer.all_special_ids)

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Filter out special tokens
        filtered_tokens = [t for t in tokens if t not in special_ids]

        for token_id in filtered_tokens:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

    # Sort by frequency and return top_k
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_tokens[:top_k])
