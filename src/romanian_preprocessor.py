"""
Romanian Text Preprocessor for Chatterbox TTS Finetuning

Instead of extending the tokenizer with new Romanian-specific tokens (which causes
posterior collapse / weak text embedding signals — see upstream issues #6, #12),
this module maps Romanian special characters to phonetically equivalent sequences
that ALREADY EXIST in the base Chatterbox vocabulary (2454 tokens).

Strategy:
  - ș/Ș (and cedilla variants ş/Ş) → "sh" (BPE token ID 120, English /ʃ/)
  - ț/Ț (and cedilla variants ţ/Ţ) → "ts" (BPE token ID 192, English /ts/)
  - ă/â/î → kept as-is (already in vocab at IDs 2413, 395, 407)
  - All text lowercased (reduces token diversity, aids generalization)

This approach:
  1. Requires NO vocabulary extension (original vocab_size = 2454)
  2. Uses only well-trained pretrained embeddings
  3. Avoids the "weak text signal" → posterior collapse problem
  4. Is phonetically accurate for Romanian pronunciation

References:
  - gokhaneraslan/chatterbox-finetuning#6  (gibberish audio, structural WTE issue)
  - gokhaneraslan/chatterbox-finetuning#12 (posterior collapse, same root cause)
  - gokhaneraslan/chatterbox-finetuning#14 (recommends standard model for quality)

Author: Adrian Stanea / ACP3
License: Apache 2.0
"""

import re
import unicodedata
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Mapping modes
# ─────────────────────────────────────────────────────────────────────────────

# Mode 1 (RECOMMENDED): Phoneme mapping for consonants, keep existing chars for vowels
# Best balance of information preservation and embedding quality.
PHONEME_MAP = {
    # Romanian ș/ş → English "sh" (/ʃ/, BPE token ID 120)
    'ș': 'sh',   # U+0219 LATIN SMALL LETTER S WITH COMMA BELOW
    'Ș': 'sh',   # U+0218 LATIN CAPITAL LETTER S WITH COMMA BELOW
    'ş': 'sh',   # U+015F LATIN SMALL LETTER S WITH CEDILLA (Turkish variant, same sound)
    'Ş': 'sh',   # U+015E LATIN CAPITAL LETTER S WITH CEDILLA

    # Romanian ț/ţ → English "ts" (/ts/, BPE token ID 192)
    'ț': 'ts',   # U+021B LATIN SMALL LETTER T WITH COMMA BELOW
    'Ț': 'ts',   # U+021A LATIN CAPITAL LETTER T WITH COMMA BELOW
    'ţ': 'ts',   # U+0163 LATIN SMALL LETTER T WITH CEDILLA (older variant)
    'Ţ': 'ts',   # U+0162 LATIN CAPITAL LETTER T WITH CEDILLA

    # Vowels: map uppercase → lowercase (chars already in vocab)
    'Ă': 'ă',    # U+0102 → U+0103 (ID 2413)
    'Â': 'â',    # U+00C2 → U+00E2 (ID 395)
    'Î': 'î',    # U+00CE → U+00EE (ID 407)
}

# Mode 2 (AGGRESSIVE): Map everything to pure ASCII
# Maximum compatibility, all tokens guaranteed well-trained,
# but loses vowel distinctions (ă/a, â/a, î/i become ambiguous).
ASCII_MAP = {
    'ș': 'sh',  'Ș': 'sh',  'ş': 'sh',  'Ş': 'sh',
    'ț': 'ts',  'Ț': 'ts',  'ţ': 'ts',  'Ţ': 'ts',
    'ă': 'a',   'Ă': 'a',
    'â': 'a',   'Â': 'a',
    'î': 'i',   'Î': 'i',
}


def preprocess_romanian(
    text: str,
    lowercase: bool = True,
    mode: str = "phoneme",
) -> str:
    """
    Preprocess Romanian text for Chatterbox TTS finetuning.

    Maps Romanian diacritical characters to sequences that already exist
    in the base Chatterbox tokenizer vocabulary, avoiding the need to
    extend the vocabulary (which causes posterior collapse).

    Args:
        text: Input Romanian text
        lowercase: Whether to lowercase the entire text (recommended)
        mode: Preprocessing mode:
            - "phoneme" (default): Map ș→sh, ț→ts, keep ă/â/î
            - "ascii": Map everything to pure ASCII (ă→a, â→a, î→i too)

    Returns:
        Preprocessed text using only existing vocabulary tokens

    Examples:
        >>> preprocess_romanian("Știință și înțelepciune.")
        'shtiiintsa shi intseleptsiune.'

        >>> preprocess_romanian("Țara mea frumoasă.")
        'tsara mea frumoasă.'

        >>> preprocess_romanian("Țara mea frumoasă.", mode="ascii")
        'tsara mea frumoasa.'
    """
    if not text:
        return text

    # Select mapping
    char_map = PHONEME_MAP if mode == "phoneme" else ASCII_MAP

    # Apply character mapping (before lowercasing, since map handles case)
    result = []
    for ch in text:
        if ch in char_map:
            result.append(char_map[ch])
        else:
            result.append(ch)
    text = ''.join(result)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Normalize Unicode: handle any remaining decomposed forms
    # NFC ensures consistent representation (e.g., a + combining breve → ă)
    text = unicodedata.normalize('NFC', text)

    return text


def punc_norm_romanian(text: str) -> str:
    """
    Punctuation normalization adapted for Romanian.

    Based on the original punc_norm from ChatterboxTTS but:
    - Does NOT capitalize the first letter (we want all lowercase)
    - Handles Romanian-specific punctuation patterns
    - Applied AFTER preprocess_romanian()
    """
    if len(text) == 0:
        return "you need to add some text for me to talk."

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/LLM punctuation
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("\u201c", "\""),   # "
        ("\u201d", "\""),   # "
        ("\u2018", "'"),    # '
        ("\u2019", "'"),    # '
        ("\u201e", "\""),   # „ (Romanian opening quote)
        ("\u201f", "\""),   # ‟
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def preprocess_and_normalize(
    text: str,
    lowercase: bool = True,
    mode: str = "phoneme",
) -> str:
    """
    Full preprocessing pipeline: Romanian char mapping + punctuation normalization.

    This replaces the standard punc_norm() for Romanian finetuning.
    Call this instead of punc_norm() in preprocess_ljspeech.py and inference.py.

    Args:
        text: Raw Romanian text
        lowercase: Whether to lowercase (default True)
        mode: "phoneme" or "ascii"

    Returns:
        Fully preprocessed text ready for tokenization
    """
    text = preprocess_romanian(text, lowercase=lowercase, mode=mode)
    text = punc_norm_romanian(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Verify all characters in text have vocab coverage
# ─────────────────────────────────────────────────────────────────────────────

def check_vocab_coverage(text: str, vocab: dict) -> dict:
    """
    Check which characters in the preprocessed text are covered by the vocabulary.

    Args:
        text: Preprocessed text
        vocab: Dictionary of token → ID from tokenizer.json

    Returns:
        Dict with 'covered', 'missing', 'coverage_pct' keys
    """
    unique_chars = set(text.replace(' ', ''))
    covered = {ch for ch in unique_chars if ch in vocab}
    missing = unique_chars - covered

    return {
        'covered': sorted(covered),
        'missing': sorted(missing),
        'coverage_pct': len(covered) / max(len(unique_chars), 1) * 100,
        'total_unique': len(unique_chars),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI: Preprocess a metadata CSV file
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_metadata_csv(
    input_path: str,
    output_path: str,
    mode: str = "phoneme",
    lowercase: bool = True,
):
    """
    Preprocess a LJSpeech-format metadata CSV, applying Romanian text mapping.

    Input format:  ID|RawText|NormText  (pipe-separated, no header)
    Output format: Same, with NormText column preprocessed.

    Args:
        input_path: Path to input metadata.csv
        output_path: Path to output preprocessed metadata.csv
        mode: "phoneme" or "ascii"
        lowercase: Whether to lowercase text
    """
    import csv

    processed_count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.reader(fin, delimiter='|', quoting=csv.QUOTE_NONE)
        writer = csv.writer(fout, delimiter='|', quoting=csv.QUOTE_NONE)

        for row in reader:
            if len(row) >= 3:
                # LJSpeech format: ID | RawText | NormText
                row[1] = preprocess_romanian(row[1], lowercase=lowercase, mode=mode)
                row[2] = preprocess_romanian(row[2], lowercase=lowercase, mode=mode)
            elif len(row) >= 2:
                row[1] = preprocess_romanian(row[1], lowercase=lowercase, mode=mode)

            writer.writerow(row)
            processed_count += 1

    return processed_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Romanian text preprocessor for Chatterbox TTS finetuning"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: preprocess text
    text_parser = subparsers.add_parser("text", help="Preprocess a single text string")
    text_parser.add_argument("input_text", help="Text to preprocess")
    text_parser.add_argument("--mode", choices=["phoneme", "ascii"], default="phoneme")
    text_parser.add_argument("--no-lowercase", action="store_true")

    # Subcommand: preprocess CSV
    csv_parser = subparsers.add_parser("csv", help="Preprocess a metadata CSV file")
    csv_parser.add_argument("input", help="Input CSV path")
    csv_parser.add_argument("output", help="Output CSV path")
    csv_parser.add_argument("--mode", choices=["phoneme", "ascii"], default="phoneme")
    csv_parser.add_argument("--no-lowercase", action="store_true")

    # Subcommand: demo
    demo_parser = subparsers.add_parser("demo", help="Show preprocessing examples")

    args = parser.parse_args()

    if args.command == "text":
        result = preprocess_romanian(
            args.input_text,
            lowercase=not args.no_lowercase,
            mode=args.mode,
        )
        print(f"Input:  {args.input_text}")
        print(f"Output: {result}")

    elif args.command == "csv":
        count = preprocess_metadata_csv(
            args.input, args.output,
            mode=args.mode,
            lowercase=not args.no_lowercase,
        )
        print(f"Preprocessed {count} rows: {args.input} → {args.output}")

    elif args.command == "demo" or args.command is None:
        # Show examples
        examples = [
            "Bună ziua, ce mai faci?",
            "Știință și înțelepciune.",
            "Țara mea frumoasă.",
            "De asemenea, contează și dacă imobilul este la stradă sau nu.",
            "Înțelepciunea înseamnă să știi că și cea mai întunecată noapte își arată strălucirea stelelor.",
        ]

        print("=" * 80)
        print("Romanian Text Preprocessor for Chatterbox TTS")
        print("=" * 80)

        for mode in ["phoneme", "ascii"]:
            print(f"\n--- Mode: {mode} ---")
            for ex in examples:
                result = preprocess_and_normalize(ex, mode=mode)
                print(f"  IN:  {ex}")
                print(f"  OUT: {result}")
                print()
