"""Perplexity calculation for language models.

Perplexity is a measure of how well a probability model predicts a sample.
For language models, it quantifies how "surprised" the model is by the text.
Lower perplexity indicates better predictive performance.

Mathematically: PPL = exp(cross_entropy_loss)
where cross_entropy_loss is the average negative log-likelihood per token.
"""

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 8,
) -> float:
    """
    Calculate average perplexity over a collection of texts.

    Perplexity measures the model's uncertainty in predicting the next token.
    It's computed as exp(loss), where loss is the cross-entropy loss.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        batch_size: Number of texts to process in each batch

    Returns:
        Average perplexity across all texts
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(texts), batch_size),
            total=num_batches,
            desc="Computing perplexity",
        ):
            batch_texts = texts[i : i + batch_size]

            # Tokenize texts
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Limit sequence length
            ).to(device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            # Create labels with proper masking: ignore padding tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore PAD tokens in loss calculation

            # Calculate loss using properly masked labels
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Average loss and convert to perplexity
    if total_tokens == 0:
        raise ValueError("No tokens found in input texts")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity
