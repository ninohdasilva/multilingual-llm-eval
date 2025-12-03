"""Compute metric rankings and generate comparative analysis."""

from models.schemas import EvaluationReport


def compute_rankings(report: EvaluationReport) -> list[dict]:
    """
    Compute rankings for each metric across languages.

    Args:
        report: EvaluationReport with results

    Returns:
        List of dictionaries with language and rank information
    """
    # Collect all metrics
    languages = []
    perplexities = []
    bpcs = []
    entropies = []
    gzip_ratios = []

    for eval_result in report.evaluations:
        languages.append(eval_result.language_info.code)
        perplexities.append(eval_result.metrics.perplexity)
        bpcs.append(eval_result.metrics.bpc)
        entropies.append(eval_result.metrics.entropy)
        gzip_ratios.append(eval_result.metrics.gzip_ratio)

    # Compute ranks (lower is better for all metrics)
    def rank_values(values: list[float], reverse: bool = False) -> list[int]:
        """Compute ranks for a list of values."""
        sorted_pairs = sorted(enumerate(values), key=lambda x: x[1], reverse=reverse)
        ranks = [0] * len(values)
        for rank, (idx, _) in enumerate(sorted_pairs, start=1):
            ranks[idx] = rank
        return ranks

    ppl_ranks = rank_values(perplexities, reverse=False)  # Lower is better
    bpc_ranks = rank_values(bpcs, reverse=False)  # Lower is better
    entropy_ranks = rank_values(entropies, reverse=False)  # Lower is better
    gzip_ranks = rank_values(gzip_ratios, reverse=False)  # Lower is better

    # Build ranking table
    rankings = []
    for i, lang in enumerate(languages):
        rankings.append(
            {
                "language": lang,
                "rank_ppl": ppl_ranks[i],
                "rank_bpc": bpc_ranks[i],
                "rank_entropy": entropy_ranks[i],
                "rank_gzip": gzip_ranks[i],
            }
        )

    return rankings


def generate_comparative_conclusion(report: EvaluationReport) -> str:
    """
    Generate automatic conclusion comparing tokenizer-dependent vs tokenizer-agnostic metrics.

    Args:
        report: EvaluationReport with results

    Returns:
        Conclusion text
    """
    rankings = compute_rankings(report)

    # Check if rankings differ between PPL and BPC
    ppl_ranks = [r["rank_ppl"] for r in rankings]
    bpc_ranks = [r["rank_bpc"] for r in rankings]

    if ppl_ranks != bpc_ranks:
        conclusion = """The metrics reveal important differences between tokenizer-dependent and 
        tokenizer-agnostic measures. Perplexity (tokenizer-dependent) and BPC (tokenizer-agnostic) 
        produce different language rankings, demonstrating that raw perplexity is not an equitable 
        metric for multilingual evaluation. Languages with higher tokenization overhead (e.g., 
        agglutinative languages) are penalized in perplexity but may show similar or better 
        performance when measured by character-based metrics like BPC."""
    else:
        conclusion = """The metrics show consistent patterns across tokenizer-dependent (perplexity) 
        and tokenizer-agnostic (BPC, gzip, entropy) measures. This suggests that tokenization 
        differences are not the primary driver of performance variation in this evaluation."""

    return conclusion
