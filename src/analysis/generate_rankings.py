"""Generate rankings CSV from metrics."""

import csv
from pathlib import Path


def generate_rankings_csv(metrics_csv_path: Path, output_path: Path) -> None:
    """
    Generate rankings CSV from metrics CSV.

    Args:
        metrics_csv_path: Path to metrics_full.csv
        output_path: Path to save rankings.csv
    """
    # Read metrics
    metrics = []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(row)

    # Group by language
    languages = {}
    for metric in metrics:
        lang = metric["language"]
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(metric)

    # Compute rankings per language
    rankings = []
    for lang, lang_metrics in languages.items():
        # Filter to main model only (not baselines) for ranking
        main_model_metrics = [
            m
            for m in lang_metrics
            if "baseline" not in m["model"].lower()
            and m["model"] not in ["gpt2", "unigram-char", "char-5gram"]
        ]

        if not main_model_metrics:
            # If no main model, use first metric
            main_model_metrics = [lang_metrics[0]]

        metric_row = main_model_metrics[0]

        # Extract metric values
        try:
            ppl = float(metric_row["mean_ppl"])
            bpc = float(metric_row["mean_bpc"])
            entropy = float(metric_row["mean_entropy_bits"])
            gzip_ratio = float(metric_row["mean_gzip_ratio"])
            loss_bits = float(metric_row["mean_loss_bits"])
        except (ValueError, KeyError):
            continue

        rankings.append(
            {
                "language": lang,
                "ppl": ppl,
                "bpc": bpc,
                "entropy": entropy,
                "gzip_ratio": gzip_ratio,
                "loss_bits": loss_bits,
            }
        )

    # Rank across all languages for each metric
    # Lower is better for all metrics
    rankings.sort(key=lambda x: x["ppl"])
    for i, rank in enumerate(rankings):
        rank["rank_ppl"] = i + 1

    rankings.sort(key=lambda x: x["bpc"])
    for i, rank in enumerate(rankings):
        rank["rank_bpc"] = i + 1

    rankings.sort(key=lambda x: x["entropy"])
    for i, rank in enumerate(rankings):
        rank["rank_entropy"] = i + 1

    rankings.sort(key=lambda x: x["gzip_ratio"])
    for i, rank in enumerate(rankings):
        rank["rank_gzip"] = i + 1

    # Compute aggregate rank (mean of all ranks)
    for rank in rankings:
        aggregate = (
            rank["rank_ppl"]
            + rank["rank_bpc"]
            + rank["rank_entropy"]
            + rank["rank_gzip"]
        ) / 4.0
        rank["aggregate_rank"] = aggregate

    # Sort by aggregate rank (tie-breaking by lower loss_bits)
    rankings.sort(key=lambda x: (x["aggregate_rank"], x["loss_bits"]))

    # Write rankings CSV
    fieldnames = [
        "language",
        "rank_ppl",
        "rank_bpc",
        "rank_entropy",
        "rank_gzip",
        "aggregate_rank",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank in rankings:
            writer.writerow(
                {
                    "language": rank["language"],
                    "rank_ppl": rank["rank_ppl"],
                    "rank_bpc": rank["rank_bpc"],
                    "rank_entropy": rank["rank_entropy"],
                    "rank_gzip": rank["rank_gzip"],
                    "aggregate_rank": f"{rank['aggregate_rank']:.2f}",
                }
            )
