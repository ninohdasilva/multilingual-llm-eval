"""Combined calculation of perplexity and entropy in a single forward pass."""

import math
from typing import List, Tuple

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def calculate_perplexity_and_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 8,
    return_per_sentence: bool = False,
) -> Tuple[float, float, float, List[float], List[float], List[float], List[float]]:
    """
    Calculate perplexity, entropy, and BPC in a single forward pass.

    This is more efficient than calling them separately since all require
    running the model forward pass.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        batch_size: Number of texts to process in each batch
        return_per_sentence: If True, also return per-sentence metrics

    Returns:
        If return_per_sentence=False: Tuple of (perplexity, entropy, bpc)
        If return_per_sentence=True: Tuple of (perplexity, entropy, bpc,
            per_sentence_ppl, per_sentence_entropy, per_sentence_bpc, per_sentence_loss_nats)
    """
    model.eval()
    total_loss_nats = 0.0
    total_entropy = 0.0
    total_tokens = 0
    total_bpc_numerator = 0.0
    total_chars = 0

    # Per-sentence metrics for bootstrap CI
    per_sentence_ppl: List[float] = []
    per_sentence_entropy: List[float] = []
    per_sentence_bpc: List[float] = []
    per_sentence_loss_nats: List[float] = []

    with torch.no_grad():
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(texts), batch_size),
            total=num_batches,
            desc="Computing perplexity & entropy",
        ):
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

            # Create labels with proper masking: ignore padding tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore PAD tokens in loss calculation

            # Get model outputs - we need both loss (for perplexity) and logits (for entropy)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss_nats = outputs.loss  # Loss is in nats (natural log)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Convert loss from nats to bits: loss_bits = loss_nats / ln(2)
            loss_bits = loss_nats.item() / math.log(2)

            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss_nats += loss_nats.item() * num_tokens
            total_tokens += num_tokens

            # Calculate entropy from logits
            # Shift logits for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Apply softmax to get probabilities
            probs = F.softmax(shift_logits, dim=-1)

            # Calculate entropy: H = -sum(p * log2(p))
            epsilon = 1e-10
            log_probs = torch.log2(probs + epsilon)
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, seq_len-1)

            # Mask out padding positions and sum
            entropy_masked = entropy * shift_mask.float()
            total_entropy += entropy_masked.sum().item()

            # Compute per-sentence metrics and BPC
            # BPC formula: (loss_nats / ln(2)) * (num_tokens / n_chars)
            # = loss_bits * (num_tokens / n_chars)
            for j, text in enumerate(batch_texts):
                sentence_mask = attention_mask[j]
                num_tokens_sentence = sentence_mask.sum().item()
                num_chars_sentence = len(text)

                if num_chars_sentence > 0 and num_tokens_sentence > 0:
                    # Per-sentence loss in nats (approximate from batch loss)
                    sentence_loss_nats = loss_nats.item()
                    sentence_loss_bits = loss_bits

                    # BPC per sentence: loss_bits * (num_tokens / n_chars)
                    sentence_bpc = sentence_loss_bits * (
                        num_tokens_sentence / num_chars_sentence
                    )
                    total_bpc_numerator += sentence_bpc * num_chars_sentence
                    total_chars += num_chars_sentence

                    # Per-sentence perplexity
                    sentence_ppl = math.exp(sentence_loss_nats)

                    # Per-sentence entropy (average over tokens)
                    sentence_entropy_mask = shift_mask[j]
                    sentence_entropy_tokens = sentence_entropy_mask.sum().item()
                    if sentence_entropy_tokens > 0:
                        sentence_entropy = (
                            entropy_masked[j].sum().item() / sentence_entropy_tokens
                        )
                    else:
                        sentence_entropy = 0.0

                    if return_per_sentence:
                        per_sentence_ppl.append(sentence_ppl)
                        per_sentence_entropy.append(sentence_entropy)
                        per_sentence_bpc.append(sentence_bpc)
                        per_sentence_loss_nats.append(sentence_loss_nats)

    if total_tokens == 0:
        raise ValueError("No tokens found in input texts")

    # Calculate perplexity from average loss in nats
    avg_loss_nats = total_loss_nats / total_tokens
    perplexity = math.exp(avg_loss_nats)

    # Calculate average entropy
    avg_entropy = total_entropy / total_tokens

    # Calculate BPC: weighted average over all sentences
    if total_chars > 0:
        bpc = total_bpc_numerator / total_chars
    else:
        bpc = 0.0

    if return_per_sentence:
        return (
            perplexity,
            avg_entropy,
            bpc,
            per_sentence_ppl,
            per_sentence_entropy,
            per_sentence_bpc,
            per_sentence_loss_nats,
        )
    else:
        return perplexity, avg_entropy, bpc
