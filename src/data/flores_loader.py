"""FLORES-200 dataset loader with language metadata."""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from models.schemas import LanguageInfo

# Language metadata: code -> (name, family, description)
LANGUAGE_METADATA: Dict[str, Tuple[str, str, str]] = {
    "fra": (
        "French",
        "Fusional",
        "French uses inflection to express grammatical relationships. Words change form (e.g., verb conjugations) to indicate tense, person, number, and gender.",
    ),
    "tur": (
        "Turkish",
        "Agglutinative",
        "Turkish builds words by concatenating morphemes. A single word can express complex meanings through a chain of suffixes.",
    ),
    "fin": (
        "Finnish",
        "Agglutinative",
        "Finnish is highly agglutinative with extensive case systems. Words can become very long through suffixation.",
    ),
    "zho_Hans": (
        "Mandarin (Simplified)",
        "Isolating",
        "Mandarin has minimal morphology. Most words are monomorphemic, and grammatical relationships are expressed through word order and particles.",
    ),
    "swa": (
        "Swahili",
        "Agglutinative",
        "Swahili uses noun classes and verb prefixes/suffixes to build complex words and express grammatical relationships.",
    ),
    "swh": (
        "Swahili",
        "Agglutinative",
        "Swahili uses noun classes and verb prefixes/suffixes to build complex words and express grammatical relationships.",
    ),
    "hin": (
        "Hindi",
        "Fusional",
        "Hindi uses inflection for case, gender, and number. Verbs conjugate for tense, aspect, and person.",
    ),
    "nno": (
        "Norwegian Nynorsk",
        "Fusional",
        "Norwegian Nynorsk uses inflection for case, gender, and number, though less extensively than Old Norse.",
    ),
}


def load_flores_sentences(language_code: str, split: str = "dev") -> List[str]:
    """
    Load sentences from FLORES-200 dataset for a specific language.

    Args:
        language_code: FLORES language code (e.g., 'fra', 'tur')
        split: Dataset split to load ('dev', 'devtest', or 'test')

    Returns:
        List of sentence strings (200 sentences for dev split)

    Raises:
        ValueError: If language code is not supported
        KeyError: If language code not found in dataset
    """
    if language_code not in LANGUAGE_METADATA:
        raise ValueError(
            f"Unsupported language code: {language_code}. "
            f"Supported codes: {list(LANGUAGE_METADATA.keys())}"
        )

    # Load FLORES-200 dataset
    # Note: We use datasets<3.0.0 which supports dataset scripts with trust_remote_code
    dataset = None
    last_error = None

    # Map our language codes to FLORES format codes
    flores_lang_map = {
        "fra": "fra_Latn",
        "tur": "tur_Latn",
        "fin": "fin_Latn",
        "zho_Hans": "zho_Hans",
        "swa": "swa_Latn",
        "hin": "hin_Deva",
        "nno": "nno_Latn",
    }
    flores_code = flores_lang_map.get(language_code, language_code)

    # Try different loading strategies
    load_strategies = [
        # Strategy 1: Load full FLORES-200 dataset (all languages)
        lambda: load_dataset(
            "facebook/flores_200", split=split, trust_remote_code=True
        ),
        # Strategy 2: Try loading without trust_remote_code (if Parquet available)
        lambda: load_dataset("facebook/flores_200", split=split),
        # Strategy 3: Try old FLORES format
        lambda: load_dataset(
            "facebook/flores", "all", split=split, trust_remote_code=True
        ),
        # Strategy 4: Try old FLORES without config
        lambda: load_dataset("facebook/flores", split=split, trust_remote_code=True),
    ]

    for strategy in load_strategies:
        try:
            dataset = strategy()
            # Check if dataset loaded successfully
            if dataset is not None:
                # Verify it has the language we need
                available_cols = list(dataset.column_names)
                if (
                    language_code in available_cols
                    or flores_code in available_cols
                    or any(
                        lang in col
                        for col in available_cols
                        for lang in [language_code, flores_code.split("_")[0]]
                    )
                ):
                    break
        except Exception as e:
            last_error = e
            continue

    if dataset is None:
        error_msg = (
            f"Failed to load FLORES dataset for language '{language_code}'.\n"
            f"Last error: {last_error}\n\n"
            f"Troubleshooting:\n"
            f"1. Ensure datasets library version < 3.0.0 is installed:\n"
            f"   uv pip install 'datasets>=2.14.0,<3.0.0'\n"
            f"2. Or manually install: pip install 'datasets<3.0.0'\n"
            f"3. Check your internet connection for dataset download\n"
        )
        raise ValueError(error_msg)

    # FLORES dataset structure - columns are prefixed with "sentence_"
    available_cols = list(dataset.column_names)

    # Determine which column contains our target language
    lang_code_to_use = None

    # Map our codes to FLORES format codes
    flores_lang_map = {
        "fra": "fra_Latn",
        "tur": "tur_Latn",
        "fin": "fin_Latn",
        "zho_Hans": "zho_Hans",
        "swa": "swh_Latn",  # FLORES uses 'swh' for Swahili
        "swh": "swh_Latn",
        "hin": "hin_Deva",
        "nno": "nno_Latn",
    }
    flores_code = flores_lang_map.get(language_code, language_code)

    # FLORES columns are prefixed with "sentence_"
    # Try different column name patterns
    candidates = [
        f"sentence_{flores_code}",  # Most common: sentence_fra_Latn
        f"sentence_{language_code}",  # Fallback: sentence_fra
        flores_code,  # Direct: fra_Latn
        language_code,  # Direct: fra
    ]

    # Handle swa -> swh mapping
    if language_code == "swa":
        candidates.insert(0, "sentence_swh_Latn")

    # Also try variations for specific languages
    if language_code == "zho_Hans":
        candidates.extend(
            [
                "sentence_zho_Hans",
                "sentence_zho_simplified",
                "sentence_zho-Hans",
                "sentence_zh-Hans",
            ]
        )

    # Search for matching column
    for candidate in candidates:
        if candidate in available_cols:
            lang_code_to_use = candidate
            break

    # If still not found, search for columns containing the language code
    if lang_code_to_use is None:
        lang_base = flores_code.split("_")[0]  # e.g., "fra" from "fra_Latn"
        for col in available_cols:
            # Check if column contains the language code (with sentence_ prefix)
            if f"sentence_{flores_code}" in col or f"sentence_{lang_base}" in col:
                lang_code_to_use = col
                break

    if lang_code_to_use is None:
        # Show first 20 columns as examples
        sample_cols = [c for c in available_cols if c.startswith("sentence_")][:20]
        raise KeyError(
            f"Language code '{language_code}' (FLORES: '{flores_code}') not found in dataset.\n"
            f"Looking for columns like: sentence_{flores_code}\n"
            f"Sample sentence columns: {sample_cols}\n"
            f"Please check the language code mapping."
        )

    # Extract sentences for this language
    sentences = dataset[lang_code_to_use]

    # Filter out None/empty sentences and convert to strings
    sentences = [str(s) for s in sentences if s is not None and str(s).strip()]

    # Apply NFC Unicode normalization and strip
    normalized_sentences = []
    for s in sentences:
        # NFC normalization, then strip
        normalized = unicodedata.normalize("NFC", s).strip()
        if normalized:  # Only keep non-empty after normalization
            normalized_sentences.append(normalized)

    return normalized_sentences


def get_language_info(language_code: str) -> LanguageInfo:
    """
    Get language metadata for a given language code.

    Args:
        language_code: FLORES language code

    Returns:
        LanguageInfo object with metadata

    Raises:
        ValueError: If language code is not supported
    """
    if language_code not in LANGUAGE_METADATA:
        raise ValueError(
            f"Unsupported language code: {language_code}. "
            f"Supported codes: {list(LANGUAGE_METADATA.keys())}"
        )

    name, family, description = LANGUAGE_METADATA[language_code]
    return LanguageInfo(
        code=language_code, name=name, family=family, description=description
    )


def load_all_languages(split: str = "dev") -> Dict[str, List[str]]:
    """
    Load FLORES sentences for all supported languages.

    Args:
        split: Dataset split to load

    Returns:
        Dictionary mapping language codes to lists of sentences
    """
    results = {}
    for lang_code in LANGUAGE_METADATA.keys():
        try:
            sentences = load_flores_sentences(lang_code, split=split)
            results[lang_code] = sentences
        except Exception as e:
            print(f"Warning: Failed to load {lang_code}: {e}")
            continue
    return results


def save_sample_sentences(
    language_code: str,
    sentences: List[str],
    output_dir: str,
    max_samples: Optional[int] = None,
) -> None:
    """
    Save sample sentences to JSONL file for reproducibility.

    Args:
        language_code: Language code
        sentences: List of sentences (will be limited to max_samples if provided)
        output_dir: Output directory
        max_samples: Maximum number of sentences to save (None = all)
    """
    output_path = Path(output_dir) / "logs" / f"sample_sentences_{language_code}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences_to_save = sentences[:max_samples] if max_samples else sentences

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences_to_save):
            json.dump(
                {"lang": language_code, "index": i, "sentence": sentence},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
