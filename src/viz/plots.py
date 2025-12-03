"""Interactive Plotly visualizations for evaluation results."""

import os

import numpy as np
import plotly.graph_objects as go

from models.schemas import EvaluationReport


def get_family_color(family: str) -> str:
    """Get color for language family."""
    color_map = {
        "Fusional": "#1f77b4",  # Blue
        "Agglutinative": "#ff7f0e",  # Orange
        "Isolating": "#2ca02c",  # Green
    }
    return color_map.get(family, "#d62728")  # Default: red


def plot_bpc_by_language(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create interactive bar chart of BPC by language.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    languages = []
    bpc_values = []
    colors = []
    families = []

    for eval_result in report.evaluations:
        lang_code = eval_result.language_info.code
        lang_name = eval_result.language_info.name
        family = eval_result.language_info.family
        bpc_val = eval_result.metrics.bpc

        languages.append(f"{lang_name}\n({lang_code})")
        bpc_values.append(bpc_val)
        families.append(family)
        colors.append(get_family_color(family))

    fig = go.Figure(
        data=[
            go.Bar(
                x=languages,
                y=bpc_values,
                marker_color=colors,
                text=[f"{v:.6f}" for v in bpc_values],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "BPC: %{y:.6f}<br>"
                    "Family: %{customdata}<extra></extra>"
                ),
                customdata=families,
            )
        ]
    )

    fig.update_layout(
        title="Bits-Per-Character (BPC) by Language",
        xaxis_title="Language",
        yaxis_title="Bits-Per-Character",
        hovermode="closest",
        template="plotly_white",
        height=500,
    )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "bpc_by_language.html"))
    fig.write_image(os.path.join(output_dir, "bpc_by_language.png"))

    return fig


def plot_tokenizer_bias(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create interactive bar chart of tokenizer bias (tokens/char ratio).

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    languages = []
    ratios = []
    colors = []
    families = []

    for eval_result in report.evaluations:
        lang_code = eval_result.language_info.code
        lang_name = eval_result.language_info.name
        family = eval_result.language_info.family
        ratio = eval_result.metrics.tokenizer_stats.tokens_per_char

        languages.append(f"{lang_name}\n({lang_code})")
        ratios.append(ratio)
        families.append(family)
        colors.append(get_family_color(family))

    fig = go.Figure(
        data=[
            go.Bar(
                x=languages,
                y=ratios,
                marker_color=colors,
                text=[f"{v:.4f}" for v in ratios],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Tokens/Char: %{y:.4f}<br>"
                    "Family: %{customdata}<extra></extra>"
                ),
                customdata=families,
            )
        ]
    )

    fig.update_layout(
        title="Tokenizer Bias: Tokens per Character Ratio",
        xaxis_title="Language",
        yaxis_title="Tokens per Character",
        hovermode="closest",
        template="plotly_white",
        height=500,
    )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "tokenizer_bias.html"))
    fig.write_image(os.path.join(output_dir, "tokenizer_bias.png"))

    return fig


def plot_ppl_vs_bpc(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create interactive scatter plot of Perplexity vs BPC.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    ppl_values = []
    bpc_values = []
    languages = []
    colors = []
    families = []

    for eval_result in report.evaluations:
        lang_code = eval_result.language_info.code
        lang_name = eval_result.language_info.name
        family = eval_result.language_info.family
        ppl = eval_result.metrics.perplexity
        bpc_val = eval_result.metrics.bpc

        ppl_values.append(ppl)
        bpc_values.append(bpc_val)
        languages.append(f"{lang_name} ({lang_code})")
        families.append(family)
        colors.append(get_family_color(family))

    # Create scatter plot
    fig = go.Figure()

    # Group by family for legend
    family_groups = {}
    for i, family in enumerate(set(families)):
        family_groups[family] = {
            "ppl": [
                ppl_values[j] for j in range(len(ppl_values)) if families[j] == family
            ],
            "bpc": [
                bpc_values[j] for j in range(len(bpc_values)) if families[j] == family
            ],
            "labels": [
                languages[j] for j in range(len(languages)) if families[j] == family
            ],
        }

    for family, data in family_groups.items():
        fig.add_trace(
            go.Scatter(
                x=data["ppl"],
                y=data["bpc"],
                mode="markers+text",
                marker=dict(size=12, color=get_family_color(family)),
                text=[label.split("(")[0].strip() for label in data["labels"]],
                textposition="top center",
                name=family,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Perplexity: %{x:.2f}<br>"
                    "BPC: %{y:.6f}<br>"
                    "Family: " + family + "<extra></extra>"
                ),
            )
        )

    # Add trend line
    if len(ppl_values) > 1:
        z = np.polyfit(ppl_values, bpc_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(ppl_values), max(ppl_values), 100)
        y_trend = p(x_trend)
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Trend",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Perplexity vs Bits-Per-Character",
        xaxis_title="Perplexity",
        yaxis_title="Bits-Per-Character (BPC)",
        hovermode="closest",
        template="plotly_white",
        height=500,
    )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "ppl_vs_bpc.html"))
    fig.write_image(os.path.join(output_dir, "ppl_vs_bpc.png"))

    return fig


def plot_metrics_heatmap(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create interactive heatmap of normalized metrics.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    languages = []
    metrics_data = {
        "Perplexity": [],
        "BPC": [],
        "Gzip Ratio": [],
        "Entropy": [],
        "Tokens/Char": [],
    }

    for eval_result in report.evaluations:
        lang_code = eval_result.language_info.code
        lang_name = eval_result.language_info.name
        languages.append(f"{lang_name}\n({lang_code})")

        metrics_data["Perplexity"].append(eval_result.metrics.perplexity)
        metrics_data["BPC"].append(eval_result.metrics.bpc)
        metrics_data["Gzip Ratio"].append(eval_result.metrics.gzip_ratio)
        metrics_data["Entropy"].append(eval_result.metrics.entropy)
        metrics_data["Tokens/Char"].append(
            eval_result.metrics.tokenizer_stats.tokens_per_char
        )

    # Normalize each metric to [0, 1] for visualization
    normalized_data = {}
    for metric_name, values in metrics_data.items():
        arr = np.array(values)
        if arr.max() - arr.min() > 0:
            normalized = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            normalized = np.zeros_like(arr)
        normalized_data[metric_name] = normalized.tolist()

    # Create heatmap data matrix
    metric_names = list(normalized_data.keys())
    z_data = [normalized_data[metric] for metric in metric_names]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=languages,
            y=metric_names,
            colorscale="Viridis",
            text=[
                [f"{metrics_data[metric][i]:.4f}" for i in range(len(languages))]
                for metric in metric_names
            ],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Language: %{x}<br>"
                "Normalized: %{z:.3f}<br>"
                "Raw: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Normalized Metrics Heatmap",
        xaxis_title="Language",
        yaxis_title="Metric",
        template="plotly_white",
        height=400,
    )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "heatmap_metrics.html"))
    fig.write_image(os.path.join(output_dir, "heatmap_metrics.png"))

    return fig


def plot_loss_distribution(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create boxplot/violin plot of loss distribution per language.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    languages = []
    losses = []
    colors = []

    for eval_result in report.evaluations:
        lang_name = eval_result.language_info.name
        family = eval_result.language_info.family

        if eval_result.per_sentence_losses:
            # Add one entry per sentence
            for loss in eval_result.per_sentence_losses:
                languages.append(lang_name)
                losses.append(loss)
                colors.append(get_family_color(family))

    fig = go.Figure()

    # Group by language for box plot
    unique_languages = sorted(set(languages))
    for lang in unique_languages:
        lang_losses = [
            loss for lang_name, loss in zip(languages, losses) if lang_name == lang
        ]
        family = next(
            e.language_info.family
            for e in report.evaluations
            if e.language_info.name == lang
        )

        fig.add_trace(
            go.Box(
                y=lang_losses,
                name=lang,
                marker_color=get_family_color(family),
                boxmean="sd",  # Show mean and standard deviation
            )
        )

    fig.update_layout(
        title="Loss Distribution by Language",
        xaxis_title="Language",
        yaxis_title="Loss",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "loss_distribution.html"))
    fig.write_image(os.path.join(output_dir, "loss_distribution.png"))

    return fig


def plot_top_tokens(report: EvaluationReport, output_dir: str) -> go.Figure:
    """
    Create histogram of top-50 token frequencies per language.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save the plot

    Returns:
        Plotly figure object
    """
    from plotly.subplots import make_subplots

    # Create subplots: one per language
    num_languages = len(report.evaluations)
    cols = 3
    rows = (num_languages + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[e.language_info.name for e in report.evaluations],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for idx, eval_result in enumerate(report.evaluations):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        if eval_result.token_frequencies:
            # Get tokenizer to decode token IDs
            # We need the tokenizer, but we don't have it here
            # For now, show token IDs
            token_ids = list(eval_result.token_frequencies.keys())
            frequencies = list(eval_result.token_frequencies.values())

            # Sort by frequency
            sorted_data = sorted(
                zip(token_ids, frequencies), key=lambda x: x[1], reverse=True
            )
            token_ids_sorted = [t[0] for t in sorted_data]
            frequencies_sorted = [f[1] for f in sorted_data]

            fig.add_trace(
                go.Bar(
                    x=[str(tid) for tid in token_ids_sorted],
                    y=frequencies_sorted,
                    name=eval_result.language_info.name,
                    marker_color=get_family_color(eval_result.language_info.family),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title_text="Top-50 Token Frequencies by Language",
        height=300 * rows,
        template="plotly_white",
    )

    # Update x-axis labels to be more readable
    for i in range(1, num_languages + 1):
        fig.update_xaxes(
            title_text="Token ID", row=(i - 1) // cols + 1, col=(i - 1) % cols + 1
        )
        fig.update_yaxes(
            title_text="Frequency", row=(i - 1) // cols + 1, col=(i - 1) % cols + 1
        )

    # Save as HTML and PNG
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "top_tokens.html"))
    fig.write_image(os.path.join(output_dir, "top_tokens.png"))

    return fig


def generate_all_plots(
    report: EvaluationReport, output_dir: str
) -> dict[str, go.Figure]:
    """
    Generate all visualization plots.

    Args:
        report: EvaluationReport with results
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to figure objects
    """
    plots = {}

    print("\nGenerating visualizations...")
    print("  Creating BPC by language plot...")
    plots["bpc"] = plot_bpc_by_language(report, output_dir)

    print("  Creating tokenizer bias plot...")
    plots["tokenizer_bias"] = plot_tokenizer_bias(report, output_dir)

    print("  Creating PPL vs BPC plot...")
    plots["ppl_vs_bpc"] = plot_ppl_vs_bpc(report, output_dir)

    print("  Creating metrics heatmap...")
    plots["heatmap"] = plot_metrics_heatmap(report, output_dir)

    print("  Creating loss distribution plot...")
    plots["loss_distribution"] = plot_loss_distribution(report, output_dir)

    print("  Creating top tokens plot...")
    plots["top_tokens"] = plot_top_tokens(report, output_dir)

    print(f"\nAll plots saved to: {output_dir}")

    return plots
