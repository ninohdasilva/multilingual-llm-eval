"""Evaluation orchestrator for multilingual LLM assessment."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.flores_loader import get_language_info, load_flores_sentences
from metrics import gzip_compression, tokenizer_stats
from metrics.combined import calculate_perplexity_and_entropy
from models.schemas import (
    EvaluationReport,
    LanguageEvaluation,
    MetricResults,
    TokenizerStats,
)


def detect_device() -> str:
    """
    Detect the best available device for model inference.

    Priority: CUDA > MPS > CPU

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Evaluator:
    """Main evaluator class for running multilingual LLM evaluation."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the evaluator with a model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'mps', 'cpu'). If None, auto-detect.
        """
        self.model_name = model_name
        self.device = device if device is not None else detect_device()

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully!")

    def _get_cache_path(
        self, language_code: str, max_sentences: Optional[int], output_dir: str
    ) -> Path:
        """Get cache file path for a language evaluation."""
        cache_dir = Path(output_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache key from model name, language, and max_sentences
        cache_key = f"{self.model_name}_{language_code}_{max_sentences or 'all'}"
        # Hash to avoid filesystem issues with special characters
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return cache_dir / f"{cache_hash}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[LanguageEvaluation]:
        """Load evaluation result from cache if it exists."""
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return LanguageEvaluation(**data)
            except Exception as e:
                print(f"  Warning: Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_path: Path, evaluation: LanguageEvaluation) -> None:
        """Save evaluation result to cache."""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(evaluation.model_dump(), f, indent=2)
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

    def evaluate_language(
        self,
        language_code: str,
        max_sentences: Optional[int] = None,
        output_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> LanguageEvaluation:
        """
        Evaluate a single language.

        Args:
            language_code: FLORES language code
            max_sentences: Maximum number of sentences to evaluate (None = all)
            output_dir: Output directory for caching (optional)
            use_cache: Whether to use cached results if available

        Returns:
            LanguageEvaluation with all metrics
        """
        print(f"\nEvaluating language: {language_code}")

        # Check cache if enabled
        if use_cache and output_dir:
            cache_path = self._get_cache_path(language_code, max_sentences, output_dir)
            cached_result = self._load_from_cache(cache_path)
            if cached_result is not None:
                print(f"  Using cached results from: {cache_path}")
                return cached_result

        # Load language info and sentences
        lang_info = get_language_info(language_code)
        sentences = load_flores_sentences(language_code)

        # Limit sentences if requested
        if max_sentences is not None:
            sentences = sentences[:max_sentences]

        print(f"  Loaded {len(sentences)} sentences")

        # Store sample sentences
        sample_sentences = sentences[:3] if len(sentences) >= 3 else sentences

        # Compute all metrics
        # Combined calculation for perplexity, entropy, and BPC (single forward pass)
        print("  Computing perplexity, entropy, and BPC (combined)...")
        ppl, entropy_value, bpc_value = calculate_perplexity_and_entropy(
            self.model, self.tokenizer, sentences, self.device
        )

        print("  Computing gzip compression ratio...")
        gzip_ratio = gzip_compression.calculate_gzip_ratio(sentences)

        print("  Computing tokenizer statistics...")
        tok_stats_dict = tokenizer_stats.calculate_tokenizer_stats(
            self.tokenizer, sentences
        )
        tok_stats = TokenizerStats(**tok_stats_dict)

        # Compute per-sentence losses and token frequencies for visualization
        print("  Computing per-sentence losses...")
        from metrics.per_sentence_metrics import (
            compute_per_sentence_losses,
            compute_token_frequencies,
        )

        per_sentence_losses = compute_per_sentence_losses(
            self.model, self.tokenizer, sentences, self.device
        )

        print("  Computing token frequencies...")
        token_frequencies = compute_token_frequencies(
            self.tokenizer, sentences, top_k=50
        )

        # Generate segmentation examples
        print("  Generating segmentation examples...")
        from analysis.segmentation_examples import generate_segmentation_examples

        segmentation_examples = generate_segmentation_examples(
            self.tokenizer, sample_sentences, max_examples=2
        )

        # Create metric results
        metrics = MetricResults(
            perplexity=ppl,
            bpc=bpc_value,
            gzip_ratio=gzip_ratio,
            entropy=entropy_value,
            tokenizer_stats=tok_stats,
        )

        print(f"  Perplexity: {ppl:.2f}")
        print(f"  BPC: {bpc_value:.6f}")
        print(f"  Gzip ratio: {gzip_ratio:.4f}")
        print(f"  Entropy: {entropy_value:.4f} bits")

        result = LanguageEvaluation(
            language_info=lang_info,
            metrics=metrics,
            sentence_count=len(sentences),
            sample_sentences=sample_sentences,
            per_sentence_losses=per_sentence_losses,
            token_frequencies=token_frequencies,
            segmentation_examples=segmentation_examples,
        )

        # Save to cache if enabled
        if use_cache and output_dir:
            cache_path = self._get_cache_path(language_code, max_sentences, output_dir)
            self._save_to_cache(cache_path, result)
            print(f"  Results cached to: {cache_path}")

        return result

    def evaluate_all_languages(
        self,
        language_codes: Optional[list[str]] = None,
        max_sentences: Optional[int] = None,
        output_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> EvaluationReport:
        """
        Evaluate all specified languages.

        Args:
            language_codes: List of language codes to evaluate.
                          If None, evaluates all supported languages.
            max_sentences: Maximum number of sentences per language (None = all)
            output_dir: Output directory for caching

        Returns:
            EvaluationReport with all results
        """
        if language_codes is None:
            # Default: all 7 languages (use swh for Swahili if swa fails)
            language_codes = ["fra", "tur", "fin", "zho_Hans", "swh", "hin", "nno"]

        evaluations = []
        for lang_code in language_codes:
            try:
                eval_result = self.evaluate_language(
                    lang_code,
                    max_sentences=max_sentences,
                    output_dir=output_dir,
                    use_cache=use_cache,
                )
                evaluations.append(eval_result)
            except Exception as e:
                print(f"Error evaluating {lang_code}: {e}")
                continue

        report = EvaluationReport(
            model_name=self.model_name,
            device=self.device,
            evaluations=evaluations,
        )

        return report

    def save_csv(self, report: EvaluationReport, output_path: str) -> None:
        """Save evaluation results to CSV (instance method)."""
        save_csv_from_report(report, output_path)


def save_csv_from_report(report: EvaluationReport, output_path: str) -> None:
    """
    Save evaluation results to CSV (standalone function).

    Args:
        report: EvaluationReport to save
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to CSV format
    rows = report.to_csv_dict()

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return

        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {output_path}")
