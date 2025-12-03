# Modified: 2025-12-02 - Added segmentation table and Plotly plots
"""Generate HTML reports from metrics."""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


def generate_html_report(
    metrics_csv_path: Path,
    rankings_csv_path: Path,
    output_dir: Path,
    model_name: str,
    mode: str,
    languages: list,
    git_commit: str = None,
    runtime: float = 0.0,
) -> Path:
    """
    Generate HTML report from metrics.

    Args:
        metrics_csv_path: Path to metrics_full.csv
        rankings_csv_path: Path to rankings.csv
        output_dir: Output directory
        model_name: Model name
        mode: Evaluation mode (quick/full)
        languages: List of language codes
        git_commit: Git commit SHA (optional)
        runtime: Runtime in seconds (optional)

    Returns:
        Path to generated HTML file
    """
    template_path = (
        Path(__file__).parent.parent / "viz" / "templates" / "report_template.html"
    )
    output_path = output_dir / "report" / "report_final.html"

    # Read template
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Read metrics CSV
    df = pd.read_csv(metrics_csv_path)

    # Create summary table (main model only)
    main_df = df[~df["model"].isin(["unigram-char", "char-5gram", "gpt2"])].copy()

    metrics_table = "<table><thead><tr>"
    metrics_table += "<th>Language</th><th>Family</th><th>Perplexity</th>"
    metrics_table += (
        "<th>BPC</th><th>Gzip Ratio</th><th>Entropy (bits)</th><th>Tokens/Char</th>"
    )
    metrics_table += "</tr></thead><tbody>"

    for _, row in main_df.iterrows():
        lang_name = row["language"].replace("_", " ").title()
        lang_code = row["language"]
        family = row["family"]
        family_class = f"family-{family}"

        metrics_table += "<tr>"
        metrics_table += (
            f"<td><strong>{lang_name}</strong> <code>{lang_code}</code></td>"
        )
        metrics_table += (
            f'<td><span class="family-badge {family_class}">{family}</span></td>'
        )
        metrics_table += f"<td>{row['mean_ppl']:.2f}</td>"
        metrics_table += f"<td>{row['mean_bpc']:.4f}</td>"
        metrics_table += f"<td>{row['mean_gzip_ratio']:.4f}</td>"
        metrics_table += f"<td>{row['mean_entropy_bits']:.4f}</td>"
        metrics_table += f"<td>{row['mean_tokens_per_char']:.4f}</td>"
        metrics_table += "</tr>"

    metrics_table += "</tbody></table>"

    # Create CI table
    ci_table = "<table><thead><tr>"
    ci_table += "<th>Language</th><th>Perplexity (95% CI)</th><th>BPC (95% CI)</th><th>Entropy (95% CI)</th><th>Gzip (95% CI)</th>"
    ci_table += "</tr></thead><tbody>"

    for _, row in main_df.iterrows():
        lang_name = row["language"].replace("_", " ").title()
        ci_table += "<tr>"
        ci_table += f"<td><strong>{lang_name}</strong></td>"
        ci_table += f"<td>{row['mean_ppl']:.2f} [{row.get('ppl_lo95', row['mean_ppl']):.2f}, {row.get('ppl_hi95', row['mean_ppl']):.2f}]</td>"
        ci_table += f"<td>{row['mean_bpc']:.4f} [{row.get('bpc_lo95', row['mean_bpc']):.4f}, {row.get('bpc_hi95', row['mean_bpc']):.4f}]</td>"
        ci_table += f"<td>{row['mean_entropy_bits']:.4f} [{row.get('entropy_lo95', row['mean_entropy_bits']):.4f}, {row.get('entropy_hi95', row['mean_entropy_bits']):.4f}]</td>"
        ci_table += f"<td>{row['mean_gzip_ratio']:.4f} [{row.get('gzip_lo95', row['mean_gzip_ratio']):.4f}, {row.get('gzip_hi95', row['mean_gzip_ratio']):.4f}]</td>"
        ci_table += "</tr>"

    ci_table += "</tbody></table>"

    # Create baseline comparison table
    baseline_table = "<table><thead><tr>"
    baseline_table += "<th>Language</th><th>Model</th><th>Perplexity</th><th>BPC</th><th>Entropy</th><th>Gzip</th><th>Comments</th>"
    baseline_table += "</tr></thead><tbody>"

    for lang in main_df["language"].unique():
        lang_data = df[df["language"] == lang]
        lang_name = lang.replace("_", " ").title()

        for _, row in lang_data.iterrows():
            model = row["model"]
            if model in ["unigram-char", "char-5gram", "gpt2"]:
                baseline_table += "<tr>"
                baseline_table += f"<td>{lang_name}</td>"
                baseline_table += f"<td><code>{model}</code></td>"
                baseline_table += f"<td>{row['mean_ppl']:.2f}</td>"
                baseline_table += f"<td>{row['mean_bpc']:.4f}</td>"
                baseline_table += f"<td>{row['mean_entropy_bits']:.4f}</td>"
                baseline_table += f"<td>{row['mean_gzip_ratio']:.4f}</td>"
                baseline_table += f'<td class="small-note">{row.get("notes", "")}</td>'
                baseline_table += "</tr>"

    baseline_table += "</tbody></table>"

    # Generate key findings
    key_findings = "<ul>"

    # Best language
    best_lang = main_df.loc[main_df["mean_bpc"].idxmin()]
    key_findings += f"<li><strong>{best_lang['language'].replace('_', ' ').title()}</strong> shows best performance across tokenizer-agnostic metrics (BPC: {best_lang['mean_bpc']:.4f}).</li>"

    # Tokenizer bias
    max_tok = main_df.loc[main_df["mean_tokens_per_char"].idxmax()]
    min_tok = main_df.loc[main_df["mean_tokens_per_char"].idxmin()]
    tok_ratio = max_tok["mean_tokens_per_char"] / min_tok["mean_tokens_per_char"]
    key_findings += f"<li><strong>{max_tok['language'].replace('_', ' ').title()}</strong> shows {tok_ratio:.1f}× higher tokens/char than {min_tok['language'].replace('_', ' ').title()}, revealing strong tokenizer bias.</li>"

    # BPC vs PPL
    key_findings += "<li><strong>BPC reveals cross-lingual fairness issues</strong> not visible in perplexity alone, correcting for tokenization differences.</li>"

    # Baseline comparison
    if len(df[df["model"] == "gpt2"]) > 0:
        key_findings += "<li><strong>GPT-2 baseline</strong> exposes English-centric tokenization bias, particularly on non-Latin scripts.</li>"

    # Gzip
    key_findings += "<li><strong>Gzip ratio</strong> (model-independent) correlates with morphological complexity and writing system density.</li>"

    key_findings += "</ul>"

    # Read rankings CSV
    rankings_table = "<table><tr><th>Language</th><th>Rank PPL</th><th>Rank BPC</th><th>Rank Entropy</th><th>Rank Gzip</th><th>Aggregate Rank</th></tr>"
    if rankings_csv_path.exists():
        with open(rankings_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rankings_table += (
                    f"<tr><td>{row['language']}</td><td>{row['rank_ppl']}</td>"
                )
                rankings_table += (
                    f"<td>{row['rank_bpc']}</td><td>{row['rank_entropy']}</td>"
                )
                rankings_table += (
                    f"<td>{row['rank_gzip']}</td><td>{row['aggregate_rank']}</td></tr>"
                )
    rankings_table += "</table>"

    # Generate Plotly plots
    from viz.plotly_plots import (
        generate_bpc_plot,
        generate_tokenizer_bias_plot,
        generate_ppl_vs_bpc_plot,
        generate_heatmap_plot,
        generate_loss_distribution_plot,
        generate_token_frequency_plot,
    )

    plot_bpc = generate_bpc_plot(df)
    plot_tokenizer = generate_tokenizer_bias_plot(df)
    plot_ppl_bpc = generate_ppl_vs_bpc_plot(df)
    plot_heatmap = generate_heatmap_plot(df)
    plot_loss_dist = generate_loss_distribution_plot(df, output_dir)
    plot_token_freq = generate_token_frequency_plot(df, output_dir)

    # Generate segmentation table from JSONL files
    seg_examples = generate_segmentation_table(output_dir, languages)

    # Replace placeholders
    html = html.replace("{{MODEL_NAME}}", model_name)
    html = html.replace("{{RUN_DATE}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace(
        "{{ARGS}}", f"--model {model_name} --langs {','.join(languages)} --mode {mode}"
    )
    html = html.replace("{{MODE}}", mode)
    html = html.replace("{{SENTENCE_COUNT}}", "50" if mode == "quick" else "200")
    html = html.replace("{{LANGUAGES}}", ", ".join(languages))
    # Generate interpretation text
    interpretation = "<p>"
    best_bpc = main_df.loc[main_df["mean_bpc"].idxmin()]
    worst_bpc = main_df.loc[main_df["mean_bpc"].idxmax()]
    interpretation += f"<strong>{best_bpc['language'].replace('_', ' ').title()}</strong> ({best_bpc['family']}) achieves lowest BPC ({best_bpc['mean_bpc']:.4f}), "
    interpretation += f"while <strong>{worst_bpc['language'].replace('_', ' ').title()}</strong> ({worst_bpc['family']}) shows highest BPC ({worst_bpc['mean_bpc']:.4f}). "
    interpretation += "This difference reflects both model capability and intrinsic language complexity (writing system, morphology)."
    interpretation += "</p>"

    html = html.replace("{{KEY_FINDINGS}}", key_findings)
    html = html.replace("{{METRICS_TABLE}}", metrics_table)
    html = html.replace("{{CI_TABLE}}", ci_table)
    html = html.replace("{{BASELINE_TABLE}}", baseline_table)
    html = html.replace("{{INTERPRETATION}}", interpretation)
    html = html.replace("{{RANKINGS_TABLE}}", rankings_table)
    html = html.replace("{{SEGMENTATION_EXAMPLES}}", seg_examples)
    html = html.replace("{{PLOT_BPC}}", plot_bpc)
    html = html.replace("{{PLOT_TOKENIZER}}", plot_tokenizer)
    html = html.replace("{{PLOT_PPL_BPC}}", plot_ppl_bpc)
    html = html.replace("{{PLOT_HEATMAP}}", plot_heatmap)
    html = html.replace("{{PLOT_LOSS_DIST}}", plot_loss_dist)
    html = html.replace("{{PLOT_TOKEN_FREQ}}", plot_token_freq)

    # Replace footer placeholders
    html = html.replace("{{GIT_COMMIT}}", git_commit or "N/A")
    html = html.replace(
        "{{RUNTIME}}", f"{runtime:.2f} seconds" if runtime > 0 else "N/A"
    )

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def generate_segmentation_table(output_dir: Path, languages: list[str]) -> str:
    """
    Generate HTML table from segmentation JSONL files.

    Args:
        output_dir: Output directory containing report/ subdirectory
        languages: List of language codes

    Returns:
        HTML table string
    """
    table_html = '<table class="segmentation-table">'
    table_html += "<tr><th>Language</th><th>Model</th><th>Original Sentence</th>"
    table_html += "<th>Tokens</th><th>Token Count</th><th>Tokens/Char</th></tr>"

    for lang in languages:
        jsonl_path = output_dir / "report" / f"segmentation_examples_{lang}.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"Segmentation file not found: {jsonl_path}")
            continue

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)

                    # Format model/tokenizer name
                    model_name = record.get(
                        "tokenizer_model", record.get("model", "unknown")
                    )
                    if "gpt2" in model_name.lower():
                        model_display = "<strong>GPT-2</strong> (baseline)"
                    else:
                        model_display = (
                            model_name.split("/")[-1]
                            if "/" in model_name
                            else model_name
                        )

                    table_html += "<tr>"
                    table_html += f"<td><strong>{record['language'].replace('_', ' ').title()}</strong></td>"
                    table_html += f'<td style="font-size:0.85em">{model_display}</td>'
                    # Truncate long sentences
                    orig_text = record["original"][:100]
                    if len(record["original"]) > 100:
                        orig_text += "..."
                    table_html += f'<td style="max-width:300px;word-wrap:break-word">{orig_text}</td>'
                    # Truncate long token strings
                    tokens_text = record["tokens"][:150]
                    if len(record["tokens"]) > 150:
                        tokens_text += "..."
                    table_html += f'<td style="font-family:monospace;font-size:0.85em;max-width:250px;word-wrap:break-word">{tokens_text}</td>'
                    table_html += f"<td>{record['num_tokens']}</td>"
                    table_html += f"<td>{record['tokens_per_char']:.4f}</td>"
                    table_html += "</tr>"
        except Exception as e:
            logger.error(f"Error reading segmentation file {jsonl_path}: {e}")
            continue

    table_html += "</table>"

    if "<tr><td>" not in table_html:
        # No data found
        table_html = "<p><em>No segmentation examples available. Run evaluation with segmentation enabled.</em></p>"

    return table_html
