"""Generate plots with 95% CI error bars."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    # seaborn is optional
    pass


def generate_plots_from_csv(metrics_csv_path: Path, output_dir: Path) -> None:
    """
    Generate all required plots from metrics CSV.

    Args:
        metrics_csv_path: Path to metrics_full.csv
        output_dir: Directory to save plots
    """
    # Read metrics
    metrics = []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(row)

    # Filter to main model only for most plots
    main_metrics = [
        m
        for m in metrics
        if m["model"] not in ["unigram-char", "char-5gram", "gpt2"]
    ]

    if not main_metrics:
        return

    # Set style
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    plt.rcParams["figure.dpi"] = 300

    # 1. BPC by language with CI
    languages = [m["language"] for m in main_metrics]
    bpc_means = [float(m["mean_bpc"]) for m in main_metrics]
    bpc_lo95 = [float(m["bpc_lo95"]) for m in main_metrics]
    bpc_hi95 = [float(m["bpc_hi95"]) for m in main_metrics]
    bpc_errors_low = [mean - lo for mean, lo in zip(bpc_means, bpc_lo95)]
    bpc_errors_high = [hi - mean for mean, hi in zip(bpc_means, bpc_hi95)]

    plt.figure(figsize=(10, 6))
    plt.bar(languages, bpc_means, yerr=[bpc_errors_low, bpc_errors_high], capsize=5)
    plt.xlabel("Language")
    plt.ylabel("Bits-per-character")
    plt.title("Bits-per-character by language (lower better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "bpc_by_language.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Tokenizer bias
    tokens_per_char = [float(m["mean_tokens_per_char"]) for m in main_metrics]
    tokens_std = [float(m["std_tokens_per_char"]) for m in main_metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(languages, tokens_per_char, yerr=tokens_std, capsize=5)
    plt.xlabel("Language")
    plt.ylabel("Tokens per character")
    plt.title("Tokenizer efficiency (tokens per character)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "tokenizer_bias.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. PPL vs BPC scatter
    ppl_means = [float(m["mean_ppl"]) for m in main_metrics]
    plt.figure(figsize=(8, 6))
    plt.scatter(ppl_means, bpc_means)
    for i, lang in enumerate(languages):
        plt.annotate(lang, (ppl_means[i], bpc_means[i]))
    plt.xscale("log")
    plt.xlabel("Perplexity (log scale)")
    plt.ylabel("Bits-per-character")
    plt.title("Perplexity vs Bits-per-character")
    # Add regression line
    z = np.polyfit(np.log(ppl_means), bpc_means, 1)
    p = np.poly1d(z)
    x_line = np.logspace(np.log10(min(ppl_means)), np.log10(max(ppl_means)), 100)
    plt.plot(x_line, p(np.log(x_line)), "r--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "ppl_vs_bpc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Heatmap of normalized metrics
    metrics_to_plot = ["mean_ppl", "mean_bpc", "mean_entropy_bits", "mean_gzip_ratio"]
    metric_names = ["PPL", "BPC", "Entropy", "Gzip Ratio"]
    data_matrix = []
    for metric in metrics_to_plot:
        values = [float(m[metric]) for m in main_metrics]
        # Normalize to [0, 1] (0 = best, 1 = worst)
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        data_matrix.append(normalized)

    plt.figure(figsize=(10, 6))
    try:
        import seaborn as sns
        sns.heatmap(
            data_matrix,
            xticklabels=languages,
            yticklabels=metric_names,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
        )
    except ImportError:
        # Fallback to matplotlib if seaborn not available
        plt.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        plt.xticks(range(len(languages)), languages)
        plt.yticks(range(len(metric_names)), metric_names)
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(languages)):
                plt.text(j, i, f"{data_matrix[i][j]:.2f}", ha="center", va="center")
    plt.title("Normalized metrics (0 best → 1 worst)")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated plots in {output_dir}")


