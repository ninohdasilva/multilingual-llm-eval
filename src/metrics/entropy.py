"""Entropy calculation from model logits.

Entropy measures the average "surprise" or uncertainty in the model's
predictions. High entropy means the model is uncertain (many possible
tokens seem equally likely). Low entropy means the model is confident
(one token is much more likely than others).

For each token position, we:
1. Extract logits from the model
2. Apply softmax to get probability distribution
3. Calculate entropy: H = -sum(p * log2(p))

This metric helps understand:
- Model confidence across languages
- Whether certain languages are harder to predict
- The relationship between entropy and perplexity
"""

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def calculate_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 8,
) -> float:
    """
    Calculate average entropy per token from model logits.

    Entropy quantifies the uncertainty in the model's probability distribution
    over the vocabulary. For each token position:
    - High entropy: model is uncertain (flat distribution)
    - Low entropy: model is confident (peaked distribution)

    This is related to perplexity but provides a different perspective:
    - Perplexity: "How many equally likely choices?"
    - Entropy: "How much uncertainty (in bits)?"

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        batch_size: Number of texts to process in each batch

    Returns:
        Average entropy per token (in bits)
    """
    model.eval()
    total_entropy = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing entropy"):
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

            # Get model outputs (logits)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Shift logits for next-token prediction
            # We want to predict token at position t+1 given tokens up to t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Apply softmax to get probabilities
            probs = F.softmax(
                shift_logits, dim=-1
            )  # (batch_size, seq_len-1, vocab_size)

            # Calculate entropy for each position: H = -sum(p * log2(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            log_probs = torch.log2(probs + epsilon)
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, seq_len-1)

            # Mask out padding positions and sum
            entropy_masked = entropy * shift_mask.float()
            total_entropy += entropy_masked.sum().item()
            total_tokens += shift_mask.sum().item()

    if total_tokens == 0:
        raise ValueError("No tokens found in input texts")

    avg_entropy = total_entropy / total_tokens
    return avg_entropy
