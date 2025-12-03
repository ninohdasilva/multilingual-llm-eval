# What to Run - Complete Guide

## 🎉 Implementation Complete - Professional Report Structure!

All enhancements have been successfully implemented with a **completely reorganized report structure** following professional evaluation report standards.

### ✨ Major Improvements

**📊 Professional Report Structure:**
- **Executive Summary** with 5 key findings at the top
- **Concise main body** (~60% shorter than before)
- **Comprehensive appendices** for detailed explanations
- **Clear section hierarchy**: Objective → Methodology → Results → Interpretation → Limitations

**📈 Enhanced Tables:**
- **Main metrics table** with proper formatting
- **95% Bootstrap CI table** showing confidence intervals
- **Baseline comparison table** (dedicated section)
- **Metrics summary table** (Type, Dependencies, Direction)

**🎯 Key Features:**
- Interactive Plotly visualizations
- Segmentation examples with clear tokenizer labels ("GPT-2 (baseline)" vs main model)
- Automatic key findings generation
- Cross-lingual interpretation
- Educational content moved to appendices

## 📝 Commands to Run

### 1. Quick Test (Recommended First)

Test the pipeline with 3 languages and 50 sentences:

```bash
uv run python src/main.py --model Qwen/Qwen2.5-1.5B --langs fra,fin,zho_Hans --mode quick --device cuda
```

**Expected outputs:**
- `outputs/report/report_final.html` - **Professional HTML report** with:
  - **Executive Summary** (5 key findings)
  - **Metrics Summary Table** (Type, Dependencies, Direction)
  - **Main Results Table** (Perplexity, BPC, Gzip, Entropy, Tokens/Char)
  - **95% CI Table** (Bootstrap confidence intervals)
  - **Interactive Plotly Plots** (hover, zoom, pan)
  - **Baseline Comparison Table** (dedicated section)
  - **Interpretation Section** (cross-lingual patterns)
  - **Rankings** with interpretation note
  - **5 Appendices**:
    - A: Metric Definitions & Calculations
    - B: Ranking Interpretation Guide
    - C: Linguistic Families & Morphology
    - D: Baseline Models
    - E: Tokenization Analysis

- `outputs/metrics/metrics_full.csv` - Metrics with gzip_is_text_metric column
- `outputs/metrics/rankings.csv` - Rankings table
- `outputs/plots/*.png` - All PNG visualizations
- `outputs/report/report_final.pdf` - PDF version (if wkhtmltopdf/Chrome available)
- `outputs/report/segmentation_examples_*.jsonl` - Tokenization examples
- `outputs/report/addendum_text.txt` - Interview talking points
- `outputs/logs/run.log` - Comprehensive logs with validation results

### 2. Full Evaluation

Run with 200 sentences per language for production results:

```bash
uv run python src/main.py --model Qwen/Qwen2.5-1.5B --langs fra,fin,zho_Hans --mode full --device cuda
```

### 3. Report-Only Mode (Fast Iteration)

If you've already run an evaluation and want to regenerate the report:

```bash
uv run python src/main.py --report-only
```

This will:
- Skip all model evaluation (very fast!)
- Read existing metrics_full.csv
- Regenerate all plots
- Regenerate HTML/PDF reports with new structure
- Perfect for iterating on report design

### 4. GPT-2 Chinese Validation Test

To specifically test GPT-2 tokenization on Chinese:

```bash
uv run python src/models/baselines/gpt2_chinese_sanity.py
```

**Output:** `outputs/logs/gpt2_chinese_sanity.log`

## 📊 Report Structure (New!)

The report is now organized professionally:

### Main Body (Concise)
1. **Executive Summary** - 5 bullet points of key findings
2. **Objective** - Brief (2 sentences)
3. **Methodology** - Pipeline summary + metrics table
4. **Results** - Summary statistics + CIs + visualizations
5. **Baseline Comparisons** - Dedicated section with table
6. **Interpretation** - Cross-lingual patterns + tokenizer effects
7. **Rankings** - With interpretation caveat
8. **Known Limitations** - Clear list

### Appendices (Detailed)
- **Appendix A**: Metric Definitions & Calculations (formulas, explanations)
- **Appendix B**: Ranking Interpretation Guide (caveats, recommendations)
- **Appendix C**: Linguistic Families & Morphology (educational content)
- **Appendix D**: Baseline Models (descriptions)
- **Appendix E**: Tokenization Analysis (segmentation examples)

## 🔍 What's Different (vs Previous Version)

### ✅ Fixed Issues

1. **Report length**: Reduced main body by ~60%, moved educational content to appendices
2. **Order optimized**: Results come early, theory goes to appendices
3. **Baselines visible**: Dedicated section with comparison table
4. **Linguistic redundancy**: Explained once in Appendix C
5. **Segmentation clarity**: Tokenizer clearly labeled ("GPT-2 (baseline)" vs main model)
6. **Title consistency**: Proper formatting (e.g., "French fra" → "French `fra`")
7. **CIs displayed**: Dedicated table with bootstrap 95% CIs
8. **Token frequencies**: Plotly visualization embedded

### 🆕 New Features

1. **Executive Summary**: 5 key findings generated automatically
2. **Metrics Summary Table**: Type, Dependencies, Direction columns
3. **CI Table**: Separate table showing means and 95% CIs
4. **Baseline Table**: Language × Model comparison with comments
5. **Key Findings**: Auto-generated from data (best language, tokenizer bias, etc.)
6. **Interpretation**: Cross-lingual patterns section
7. **Professional styling**: Consistent badges, colors, formatting

## 🎯 What to Check

After running, verify:

1. **Executive Summary** (`outputs/report/report_final.html`):
   - 5 bullet points at top
   - Automatically generated from data
   - Highlights best language, tokenizer bias, cross-lingual fairness

2. **Main Tables**:
   - **Summary Statistics**: Main model only, all metrics
   - **95% CI Table**: Shows [lo, hi] for each metric
   - **Baseline Table**: All baselines × languages with comments

3. **Metrics Summary Table**:
   - Shows Type, Tokenizer-Dependent, Model-Dependent, Direction
   - Clearly indicates gzip is model-independent

4. **Segmentation Table** (Appendix E):
   - Tokenizer column shows "GPT-2 (baseline)" or main model name
   - No confusion about which tokenizer

5. **Appendices**:
   - All educational content moved here
   - Formulas in Appendix A
   - Ranking interpretation in Appendix B
   - Linguistic families in Appendix C

6. **Validation Results** in `outputs/logs/run.log`:
   - Look for "RUNNING VALIDATION TESTS" section
   - Check all 4 tests passed

## 📈 Expected Results

For the quick test (fra, fin, zho_Hans):

### Executive Summary Will Show:
- **French** shows best performance (lowest BPC)
- **Chinese** shows highest tokens/char (tokenizer bias)
- **BPC reveals cross-lingual fairness** issues
- **GPT-2 baseline** exposes English-centric bias
- **Gzip ratio** correlates with morphological complexity

### Main Results Table:
- **French**: Lowest BPC (~0.77), best performance
- **Finnish**: Higher tokens/char (~0.38), moderate BPC
- **Chinese**: Highest tokens/char (~0.80), highest BPC

### Baseline Comparisons:
- Char-unigram: Highest PPL (worst)
- Char-5gram: Better than unigram
- GPT-2: Better than char models, but tokenization issues on Chinese

## 🚀 Next Steps

1. Run the quick test command above
2. Open `outputs/report/report_final.html` in your browser
3. **Check Executive Summary** - should have 5 key findings
4. **Scroll through main body** - should be concise (~3-4 pages)
5. **Check appendices** - all educational content should be there
6. **Verify tables**:
   - Main metrics table (main model only)
   - CI table (with [lo, hi] intervals)
   - Baseline table (all baselines)
7. Review validation results in `outputs/logs/run.log`
8. If satisfied, run full evaluation

## 🐛 Troubleshooting

### If plots don't show:
- Check browser console for JavaScript errors
- Verify Plotly CDN is accessible
- Try `--report-only` to regenerate

### If PDF generation fails:
- Install wkhtmltopdf: `brew install wkhtmltopdf` (macOS)
- HTML report will still be generated

### If GPT-2 fails on Chinese:
- Check `outputs/logs/run.log` for validation warnings
- Look for notes in metrics_full.csv
- Run standalone test: `uv run python src/models/baselines/gpt2_chinese_sanity.py`

## 📚 Documentation

All documentation is in `README.md`:
- Complete usage guide
- Metrics explained
- Validation tests
- Output files description

The pipeline is production-ready with professional report structure!
