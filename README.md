# Multilingual LLM Evaluation Pipeline

A reproducible evaluation framework for assessing multilingual language models across diverse language families using FLORES-200 dataset. This project implements comprehensive metrics (perplexity, BPC, compression, entropy) with 4 baseline models and generates visualizations with bootstrap confidence intervals.

## 🎯 Project Overview

This pipeline evaluates language models on multiple languages from different morphological families (Fusional, Agglutinative, Isolating) to understand:

- **Model performance** across languages
- **Tokenizer bias** and its impact on metrics
- **Language family effects** on evaluation
- **Fair cross-lingual comparison** using tokenizer-neutral metrics

### Key Features

- ✅ **5 Comprehensive Metrics**: Perplexity, Bits-Per-Character (BPC), Gzip Compression, Entropy, Tokenizer Statistics
- ✅ **3 Baseline Models**: Char-unigram, Char-5gram, GPT-2 for comparison
- ✅ **Bootstrap Confidence Intervals**: 95% CI with 1000 resamples for all main metrics
- ✅ **7 Languages**: French, Turkish, Finnish, Mandarin, Swahili, Hindi, Norwegian Nynorsk
- ✅ **Visualizations**: PNG plots with 95% CI error bars (300 DPI)
- ✅ **Educational HTML/PDF Report**: Comprehensive explanations with Known Limitations section
- ✅ **OOM Handling**: Automatic batch size reduction on out-of-memory errors
- ✅ **Comprehensive Logging**: Detailed logs with timestamps, errors, and git commit SHA
- ✅ **Reproducible**: Fixed random seed (42) for all operations

## 📋 Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Optional: wkhtmltopdf or Chrome for PDF generation

## 🚀 Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## 💻 Usage

### Quick Evaluation (Recommended for Testing)

Run a quick test with 3 languages and 50 sentences per language:

```bash
uv run python src/main.py --model HuggingFaceTB/SmolLM3-3B --langs fra,fin,zho_Hans --mode quick --device cuda
```

### Full Evaluation

Run full evaluation with 3 languages and 200 sentences per language:

```bash
uv run python src/main.py --model HuggingFaceTB/SmolLM3-3B --langs fra,fin,zho_Hans --mode full --device cuda
```

### Using the Evaluation Script

Run the provided evaluation script for quick and full runs:

```bash
./run.sh
```

This runs:
1. Quick 3-language test (fra, fin, zho_Hans) with 50 sentences
2. Full 3-language evaluation with 200 sentences per language

### CLI Options

**Required Arguments (unless using --report-only):**
- `--model MODEL_NAME`: HuggingFace model identifier (e.g., HuggingFaceTB/SmolLM3-3B)
- `--langs LANG1,LANG2,...`: Comma-separated FLORES language codes (e.g., fra,fin,zho_Hans)

**Optional Arguments:**
- `--mode MODE`: Evaluation mode - `full` (200 sentences) or `quick` (50 sentences). Default: `full`
- `--quick`: Alias for `--mode quick` (overrides --mode if specified)
- `--device DEVICE`: Device to use - `cuda` or `cpu`. If not specified, auto-detects (cuda if available, else cpu)
- `--report-only`: Skip evaluation and regenerate plots/reports from existing `metrics_full.csv`. Useful for quick iteration on visualizations without re-running models.

### Report-Only Mode (Quick Iteration)

If you've already run an evaluation and want to regenerate plots/reports without re-running models:

```bash
uv run python src/main.py --report-only
```

This will:
- Read existing `outputs/metrics/metrics_full.csv`
- Regenerate all plots (if missing or outdated)
- Regenerate HTML and PDF reports
- Skip all model evaluation (much faster)

**Note:** Model and language information will be inferred from the CSV if not provided.

### Supported Language Codes

- `fra` - French
- `tur` - Turkish
- `fin` - Finnish
- `zho_Hans` - Mandarin (Simplified)
- `swh` - Swahili
- `hin` - Hindi
- `nno` - Norwegian Nynorsk

## 📊 Metrics Explained

### Token-Based Metrics (Tokenizer-Dependent)

- **Perplexity (PPL)**: Measures model surprise. Lower is better. Computed as `exp(cross-entropy loss)`. **Highly dependent on tokenization** - languages with more tokens per character naturally have higher perplexity.
- **Entropy (bits)**: Average uncertainty in next-token predictions. Computed from softmax distribution over vocabulary. Lower is better.

### Tokenizer-Agnostic Metrics

- **Bits-per-Character (BPC)**: Normalizes loss by character count. Formula: `(loss_nats / ln(2)) × (tokens / chars)`. Enables fair cross-language comparison by removing tokenizer bias. Lower is better.
- **Gzip Compression Ratio**: **Text-only metric (model-independent)**. Measures source text compressibility using gzip (level 6). Formula: `compressed_size / original_size`. Characterizes intrinsic text complexity. Lower values = more compressible/redundant text. **This metric is identical across all models for the same language.**

### Why BPC Matters

BPC is crucial for multilingual evaluation because it removes tokenizer bias. Languages with complex morphology (Finnish, Turkish) require more tokens per character, artificially inflating token-based metrics like perplexity. BPC normalizes for this, revealing true model understanding.

**Example:** If Finnish requires 2× more tokens than English for the same text, its perplexity will be artificially higher even if the model understands both equally well. BPC corrects for this.

### Validation Tests

The pipeline automatically runs comprehensive validation tests including:

1. **Token Round-trip Verification**: Tests whether each tokenizer can properly encode and decode text for each language
2. **PPL Sanity Checks**: Ensures perplexity values are >1.0 and finite
3. **Gzip Consistency**: Verifies gzip ratios are identical across models (since it's text-only)
4. **Segmentation File Integrity**: Checks for UTF-8 encoding artifacts

### Running GPT-2 Chinese Sanity Test

To diagnose GPT-2 tokenization issues on Chinese separately:

```bash
uv run python src/models/baselines/gpt2_chinese_sanity.py --output outputs/logs/gpt2_chinese_sanity.log
```

This test validates whether GPT-2's tokenizer can properly handle Chinese characters through encode/decode round-trips.

## 📊 Output Files

After running the evaluation, you'll find:

### `outputs/metrics/metrics_full.csv`

CSV file with all computed metrics for each language and model (main model + 3 baselines):
- `language`: Language code (e.g., "fra", "tur")
- `model`: Model identifier (main model or baseline: unigram-char, char-5gram, gpt2)
- `family`: Morphological family (Fusional/Agglutinative/Isolating)
- `sentences_evaluated`: Number of sentences evaluated
- `mean_ppl`, `std_ppl`, `ppl_lo95`, `ppl_hi95`: Perplexity with bootstrap 95% CI
- `mean_loss_nats`, `mean_loss_bits`: Loss in nats and bits
- `mean_bpc`, `std_bpc`, `bpc_lo95`, `bpc_hi95`: Bits-per-character with bootstrap 95% CI
- `mean_entropy_bits`, `std_entropy_bits`, `entropy_lo95`, `entropy_hi95`: Entropy with bootstrap 95% CI
- `mean_gzip_ratio`, `std_gzip_ratio`, `gzip_lo95`, `gzip_hi95`: Gzip compression ratio with bootstrap 95% CI
- `gzip_is_text_metric`: Boolean flag (always "true") indicating gzip is model-independent
- `mean_tokens_per_char`, `std_tokens_per_char`: Tokenizer statistics
- `notes`: Additional notes (e.g., "gpt2_tokenizer_roundtrip_failed_for_zho_Hans")

### `outputs/metrics/rankings.csv`

Rankings table with per-metric ranks and aggregate rank:
- `language`: Language code
- `rank_ppl`, `rank_bpc`, `rank_entropy`, `rank_gzip`: Ranks for each metric (1 = best)
- `aggregate_rank`: Mean rank across all metrics (lower = better)

### `outputs/plots/`

PNG visualizations (300 DPI):

1. **`bpc_by_language.png`**: Bits-per-character by language with 95% CI error bars
2. **`tokenizer_bias.png`**: Tokens per character ratio with error bars
3. **`ppl_vs_bpc.png`**: Perplexity vs BPC scatter plot with regression line
4. **`heatmap_metrics.png`**: Normalized metrics heatmap (0 = best, 1 = worst)
5. **`loss_distribution.png`**: Per-sentence loss distribution boxplot per language
6. **`token_freqs_LANG.png`**: Top-50 token frequency charts (one per language)

### `outputs/report/`

Report files:

1. **`report_final.html`**: Comprehensive HTML report with:
   - Detailed methodology and metric explanations
   - Calculation formulas
   - Metrics table and visualizations
   - Per-language segmentation examples (2 sentences × models)
   - Rankings with interpretation addendum
   - Comprehensive definitions section
   - Known limitations

2. **`report_final.pdf`**: PDF version of the HTML report (requires wkhtmltopdf or Chrome)

3. **`segmentation_examples_LANG.jsonl`**: Tokenization examples showing how each tokenizer segments text (2 sentences per language × models). Each line is a JSON object with:
   - `language`: Language code
   - `model`: Model identifier
   - `original`: Original sentence
   - `tokens`: Pipe-separated tokens
   - `token_ids`: Comma-separated token IDs
   - `num_tokens`, `num_chars`, `tokens_per_char`: Statistics

4. **`addendum_text.txt`**: Key points for ranking interpretation (useful for interviews/presentations)

### `outputs/logs/`

Log files:

1. **`run.log`**: Comprehensive execution log with timestamps, model status, OOM events, validation results, and git commit SHA
2. **`sample_sentences_LANG.jsonl`**: Sample sentences used for evaluation (for reproducibility)
3. **`gpt2_chinese_sanity.log`**: GPT-2 Chinese tokenization validation results (if run separately)
6. **`token_freqs_LANG.png`**: Top-50 token frequency bar chart per language (one per language)
7. **`token_freqs_LANG.csv`**: Top-50 token frequencies CSV per language

### `outputs/report/report_final.html`

Comprehensive educational HTML report with:
- **Header**: Model name, run date, arguments used
- **Objective & Methodology**: Experiment goals and approach
- **Metrics Overview**: Definitions and directions (lower vs higher is better)
- **Summary Statistics**: Complete metrics table for all languages and models
- **Visualizations**: All plots embedded as PNG images
- **Per-Language Segmentation Analysis**: Tokenization examples
- **Rankings Table**: Comparative rankings with aggregate rank
- **Known Limitations**: Verbatim limitations section as specified

### `outputs/report/report_final.pdf`

PDF version of the report (generated if wkhtmltopdf or Chrome is available)

### `outputs/logs/run.log`

Comprehensive log file with:
- Timestamps for all operations
- Command-line arguments
- Model loading status
- Batch sizes used and OOM retries
- CPU fallback events
- Exceptions with tracebacks
- Git commit SHA (if available)

### `outputs/logs/REPORT_SUMMARY.txt`

Summary of the evaluation run:
- Run timestamp
- Command executed
- Models and languages evaluated
- Sentence counts
- Output locations
- Duration
- Deviations from spec

### `outputs/logs/sample_sentences_LANG.jsonl`

Sample sentences used for each language (JSONL format) for reproducibility

## 📐 Metrics Explained

### 1. Perplexity

**What it measures**: How "surprised" the model is by the text.

**Formula**: `PPL = exp(cross_entropy_loss)`

**Computation**:
- Uses proper attention masking: `labels[attention_mask == 0] = -100` to ignore padding tokens
- Computed from model loss with masked labels
- Only non-padding tokens contribute to the loss

**Interpretation**:
- **Lower is better** (model is less surprised)
- A perplexity of 10 means the model is as uncertain as choosing from 10 equally likely tokens
- **Limitation**: Token-based, affected by tokenizer choices
- **Expected range**: For a 3B model on FLORES-200, typically 5-80

### 2. Bits-Per-Character (BPC)

**What it measures**: Tokenizer-neutral metric for fair cross-lingual comparison.

**Formula**: `BPC = (loss_nats / ln(2)) * (num_tokens / num_characters)` (computed per sentence, then aggregated)

**Why it matters**:
- Normalizes by characters instead of tokens
- Removes tokenizer bias
- Essential for comparing morphologically-rich languages (Turkish, Finnish) with isolating languages (Mandarin)
- Computed directly from loss, making it 100% tokenizer-agnostic

**Interpretation**: **Lower is better** - fewer bits needed to encode each character

### 3. Gzip Compression Ratio

**What it measures**: Information density and redundancy of text.

**Formula**: `ratio = compressed_size / original_size`

**Interpretation**:
- **Lower is better** - More compressible (more redundancy and predictable patterns)
- Higher ratio = Less compressible (high information density)
- Model-independent baseline metric

**Insight**: Languages with complex morphology often compress less well

### 4. Entropy

**What it measures**: Average uncertainty in the model's probability distribution.

**Formula**: `H = -Σ(p × log₂(p))` for each token position

**Interpretation**:
- **Lower is better** - Model is more confident (peaked distribution)
- High entropy = Model is uncertain (flat distribution)
- Measured in bits per token

### 5. Tokenizer Statistics

**What it measures**: How the tokenizer segments text differently across languages.

**Metrics**:
- `avg_tokens`: Average tokens per sentence (excluding special tokens)
- `tokens_per_char`: Tokenization efficiency ratio (excluding special tokens)
- `avg_char_length`: Average sentence length in characters

**Computation**:
- Special tokens (BOS, EOS, PAD, etc.) are excluded from token counts
- Only actual content tokens are counted for fair comparison

**Why it matters**:
- Reveals tokenizer bias
- Morphologically-rich languages often require more tokens per character
- Helps explain differences in token-based metrics
- **Lower tokens_per_char is better** - more efficient tokenization

## 🌐 Languages Evaluated

| Code | Language | Family | Morphology Type |
|------|----------|--------|-----------------|
| `fra` | French | Fusional | Uses inflection for grammar |
| `tur` | Turkish | Agglutinative | Concatenates morphemes |
| `fin` | Finnish | Agglutinative | Extensive case systems |
| `zho_Hans` | Mandarin (Simplified) | Isolating | Minimal morphology |
| `swh` | Swahili | Agglutinative | Noun classes and verb affixes |
| `hin` | Hindi | Fusional | Inflection for case/gender/number |
| `nno` | Norwegian Nynorsk | Fusional | Moderate inflection |

### Language Families

**Fusional Languages** (French, Hindi, Norwegian):
- Use inflection to express grammatical relationships
- Words change form (verb conjugations) to indicate tense, person, number, gender
- Multiple grammatical features fused into single morphemes

**Agglutinative Languages** (Turkish, Finnish, Swahili):
- Build words by concatenating morphemes
- Single words can express complex meanings through suffix chains
- Each morpheme typically has a single grammatical function

**Isolating Languages** (Mandarin):
- Minimal morphology
- Most words are monomorphemic
- Grammatical relationships expressed through word order and particles

## 🏗️ Project Structure

```
multilingual-llm-eval/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── cli.py                  # Command-line argument parsing
│   ├── config.py               # Global configuration and seeds
│   ├── models/
│   │   ├── schemas.py          # Pydantic data models
│   │   └── baselines/
│   │       ├── unigram.py      # Char-unigram baseline
│   │       ├── char_ngram.py   # Char 5-gram baseline
│   │       └── hf_transformer_eval.py # GPT-2 and Qwen baselines
│   ├── data/
│   │   └── flores_loader.py    # FLORES-200 dataset loader with NFC normalization
│   ├── metrics/
│   │   ├── perplexity.py       # Perplexity calculation
│   │   ├── bpc.py              # Bits-per-character
│   │   ├── combined.py         # Combined perplexity/entropy/BPC
│   │   ├── gzip_compression.py  # Compression ratio
│   │   ├── entropy.py          # Entropy from logits
│   │   ├── tokenizer_stats.py  # Tokenizer statistics
│   │   ├── per_sentence_metrics.py # Per-sentence losses and token frequencies
│   │   └── bootstrap_ci.py     # Bootstrap confidence intervals
│   ├── analysis/
│   │   ├── rankings.py         # Metric rankings
│   │   ├── generate_rankings.py # Rankings CSV generation
│   │   └── segmentation_examples.py # Tokenization examples
│   ├── evaluation/
│   │   ├── evaluator.py        # Main evaluation orchestrator
│   │   └── oom_handler.py      # OOM handling with batch size reduction
│   ├── viz/
│   │   ├── plots.py            # Plotly visualizations (legacy)
│   │   ├── generate_plots.py  # Plot generation with CI
│   │   └── templates/
│   │       └── report_template.html # HTML report template
│   └── report/
│       ├── report_generator.py  # HTML report generation (legacy)
│       └── generate_report.py  # HTML and PDF report generation
├── outputs/
│   ├── logs/
│   │   ├── run.log            # Comprehensive execution log
│   │   ├── REPORT_SUMMARY.txt # Run summary
│   │   └── sample_sentences_LANG.jsonl # Sample sentences per language
│   ├── metrics/
│   │   ├── metrics_full.csv   # Complete metrics with CI
│   │   └── rankings.csv       # Rankings table
│   ├── plots/                 # PNG visualizations (300 DPI)
│   └── report/
│       ├── report_final.html  # HTML report
│       └── report_final.pdf   # PDF report (if available)
├── pyproject.toml             # Project configuration
├── requirements.txt           # Pip requirements
├── run.sh                     # Example execution script
└── README.md                  # This file
```

## 🔧 Technical Details

### Device Selection

The pipeline automatically detects and uses the best available device:
1. **CUDA** (if available) - NVIDIA GPUs
2. **MPS** (if available) - Apple Silicon GPUs
3. **CPU** (fallback)

### Dataset

- **Source**: [FLORES-200](https://github.com/facebookresearch/flores) (Facebook Research)
- **Split**: Development set (`dev`)
- **Size**: 200 parallel sentences per language (configurable via --mode quick/full)
- **Format**: Raw text with NFC Unicode normalization
- **Truncation**: Only at tokenizer level (max_length=512), never at raw text level

### Model Requirements

The model must be:
- A causal language model (AutoModelForCausalLM)
- Compatible with HuggingFace Transformers
- Capable of computing loss with `labels` parameter

### Baseline Models

1. **Char-Unigram**: Character-level unigram model with epsilon smoothing (1e-12)
2. **Char-5gram**: Character-level 5-gram model with Laplace smoothing (alpha=1)
3. **GPT-2**: HuggingFace `gpt2` model (non-distilled)

### Bootstrap Confidence Intervals

All main metrics (PPL, BPC, Entropy, Gzip) include:
- Mean and standard deviation
- 95% confidence intervals (lo95, hi95) computed via bootstrap with 1000 resamples

## 🐛 Troubleshooting

### Out of Memory

If you encounter OOM errors:
- The pipeline automatically reduces batch size and retries (up to 3 times)
- Minimum batch size is 1
- Check `outputs/logs/run.log` for OOM retry messages
- For very large models, consider using CPU or a smaller model

### Missing Dependencies

If imports fail:
```bash
uv sync
```

### FLORES Dataset Download

The first run will download FLORES-200 (~500MB). Ensure you have:
- Internet connection
- Sufficient disk space
- Write permissions

### PDF Generation

If PDF generation fails:
- Install `wkhtmltopdf`: `brew install wkhtmltopdf` (macOS) or `apt-get install wkhtmltopdf` (Linux)
- Or ensure Chrome/Chromium is available in PATH
- The HTML report will still be generated

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **FLORES-200**: Facebook Research for the parallel dataset
- **HuggingFace**: For Transformers library and model hosting
- **Plotly/Matplotlib**: For visualizations

## 📖 Further Reading

- [FLORES-200 Paper](https://arxiv.org/abs/2207.04672)
- [Perplexity in NLP](https://en.wikipedia.org/wiki/Perplexity)
- [Language Morphology Types](https://en.wikipedia.org/wiki/Morphological_typology)
- [Bits-Per-Character](https://en.wikipedia.org/wiki/Entropy_(information_theory))

---

**Happy Evaluating!** 🚀

For questions or issues, please refer to the comprehensive HTML report generated in `outputs/report/report_final.html`.
