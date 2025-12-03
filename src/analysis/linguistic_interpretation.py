"""Generate linguistic interpretations based on typological families and metrics."""

from models.schemas import EvaluationReport, LanguageEvaluation


def generate_family_interpretation(
    family: str, evaluations: list[LanguageEvaluation]
) -> str:
    """
    Generate interpretation text for a typological family.

    Args:
        family: Language family (Fusional, Agglutinative, or Isolating)
        evaluations: List of language evaluations in this family

    Returns:
        Interpretation text explaining expected patterns
    """
    if family == "Fusional":
        return """Fusional languages (e.g., French, Hindi, Norwegian Nynorsk) use inflection to express 
        grammatical relationships. Words change form (e.g., verb conjugations) to indicate tense, person, 
        number, and gender. This morphological ambiguity can penalize perplexity because the model must 
        learn complex inflectional paradigms. Tokenizers may struggle with highly inflected forms if they 
        are rare, potentially leading to more subword tokens."""

    elif family == "Agglutinative":
        return """Agglutinative languages (e.g., Turkish, Finnish, Swahili) build words by concatenating 
        morphemes. A single word can express complex meanings through a chain of suffixes. This leads to 
        long BPE segmentations, increasing tokens/char and artificially inflating perplexity. The tokenizer 
        often breaks these long words into many subword units, making token-based metrics appear worse even 
        if the model's underlying understanding is good."""

    elif family == "Isolating":
        return """Isolating languages (e.g., Mandarin Simplified) have minimal morphology. Most words 
        are monomorphemic, and grammatical relationships are expressed through word order and particles. 
        This leads to tokens roughly tracking characters, resulting in more stable perplexity values. 
        However, BPC may be higher due to the character-level information density of logographic or 
        ideographic writing systems."""

    return ""


def generate_all_family_interpretations(report: EvaluationReport) -> dict[str, str]:
    """
    Generate interpretations for all language families in the report.

    Args:
        report: EvaluationReport with results

    Returns:
        Dictionary mapping family name to interpretation text
    """
    families = {}
    for eval_result in report.evaluations:
        family = eval_result.language_info.family
        if family not in families:
            # Get all languages in this family
            family_evaluations = [
                e for e in report.evaluations if e.language_info.family == family
            ]
            families[family] = generate_family_interpretation(
                family, family_evaluations
            )

    return families
