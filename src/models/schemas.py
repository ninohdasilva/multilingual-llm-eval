"""Pydantic data models for type-safe evaluation results."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class LanguageInfo(BaseModel):
    """Information about a language being evaluated."""

    code: str = Field(..., description="FLORES language code (e.g., 'fra', 'tur')")
    name: str = Field(..., description="Full language name")
    family: str = Field(
        ..., description="Morphological family: Fusional, Agglutinative, or Isolating"
    )
    description: Optional[str] = Field(
        None, description="Brief description of the language's characteristics"
    )

    @field_validator("family")
    @classmethod
    def validate_family(cls, v: str) -> str:
        """Validate that family is one of the accepted types."""
        valid_families = {"Fusional", "Agglutinative", "Isolating"}
        if v not in valid_families:
            raise ValueError(f"Family must be one of {valid_families}")
        return v


class TokenizerStats(BaseModel):
    """Statistics about tokenization for a language."""

    avg_tokens: float = Field(
        ..., ge=0, description="Average number of tokens per sentence"
    )
    tokens_per_char: float = Field(
        ..., ge=0, description="Average ratio of tokens to characters"
    )
    avg_char_length: float = Field(
        ..., ge=0, description="Average sentence length in characters"
    )


class MetricResults(BaseModel):
    """All computed metrics for a language."""

    perplexity: float = Field(
        ..., gt=0, description="Perplexity score (lower is better)"
    )
    bpc: float = Field(
        ..., ge=0, description="Bits-per-character (tokenizer-neutral metric)"
    )
    gzip_ratio: float = Field(
        ..., ge=0, description="Gzip compression ratio (compressed/original)"
    )
    entropy: float = Field(..., ge=0, description="Average entropy per token (in bits)")
    tokenizer_stats: TokenizerStats = Field(..., description="Tokenizer statistics")


class LanguageEvaluation(BaseModel):
    """Complete evaluation results for a single language."""

    language_info: LanguageInfo = Field(..., description="Language metadata")
    metrics: MetricResults = Field(..., description="Computed metrics")
    sentence_count: int = Field(..., ge=0, description="Number of sentences evaluated")
    sample_sentences: List[str] = Field(
        default_factory=list, description="Sample sentences from the dataset"
    )
    per_sentence_losses: Optional[List[float]] = Field(
        default=None,
        description="Loss value for each sentence (for distribution plots)",
    )
    token_frequencies: Optional[Dict[int, int]] = Field(
        default=None, description="Token ID -> frequency mapping (for token analysis)"
    )
    segmentation_examples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Segmentation examples showing original sentence, tokens, and metrics",
    )


class EvaluationReport(BaseModel):
    """Complete evaluation report for all languages."""

    model_name: str = Field(..., description="Name of the evaluated model")
    device: str = Field(..., description="Device used for evaluation (cuda/mps/cpu)")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the evaluation was run"
    )
    evaluations: List[LanguageEvaluation] = Field(
        ..., description="Evaluation results for each language"
    )

    def to_csv_dict(self) -> List[dict]:
        """Convert report to list of dictionaries for CSV export."""
        rows = []
        for eval_result in self.evaluations:
            row = {
                "language": eval_result.language_info.code,
                "language_name": eval_result.language_info.name,
                "family": eval_result.language_info.family,
                "perplexity": eval_result.metrics.perplexity,
                "bpc": eval_result.metrics.bpc,
                "gzip_ratio": eval_result.metrics.gzip_ratio,
                "entropy": eval_result.metrics.entropy,
                "avg_tokens": eval_result.metrics.tokenizer_stats.avg_tokens,
                "tokens_per_char": eval_result.metrics.tokenizer_stats.tokens_per_char,
                "avg_char_length": eval_result.metrics.tokenizer_stats.avg_char_length,
                "sentence_count": eval_result.sentence_count,
            }
            rows.append(row)
        return rows
