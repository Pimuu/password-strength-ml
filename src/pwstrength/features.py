"""Per-password feature extraction.

Features fall into three groups, so the ablation study can add them in steps:

* COMPOSITION: length and character-class counts/fractions (what classic
  "LUDS" password meters look at).
* ENTROPY: Shannon entropy of the character distribution and "pool entropy"
  (length x log2 alphabet size, the naive NIST-style estimate).
* PATTERN: weakness indicators such as dictionary words, leetspeak, keyboard
  walks, sequences, repetition, years/dates, common-password bases and
  human-style structure (capitalised first letter, digits at the end).

The labeler's own guess estimate (``guess_bits``) is NOT a feature. Including
it would leak the label.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .patterns import (char_class, date_matches, deleet, dictionary_matches,
                       pool_size, repeat_block_spans, repeat_char_matches,
                       sequence_matches, spatial_matches)
from .wordlists import COMMON_RANK, ENGLISH_WORDS, NAMES

COMPOSITION_FEATURES = [
    "length", "n_lower", "n_upper", "n_digit", "n_symbol", "n_char_classes",
    "frac_upper", "frac_digit", "frac_symbol",
]
ENTROPY_FEATURES = [
    "shannon_entropy", "shannon_total_bits", "pool_entropy_bits",
    "unique_char_ratio", "class_transitions",
]
PATTERN_FEATURES = [
    "has_dict_word", "longest_dict_word", "dict_coverage", "n_dict_words",
    "has_leet", "longest_keyboard_run", "longest_sequence", "max_char_repeat",
    "has_block_repeat", "has_year", "has_date", "is_common_password",
    "common_base", "word_then_digits", "capitalized_first_only",
    "digits_at_end", "symbol_at_end",
]
ALL_FEATURES = COMPOSITION_FEATURES + ENTROPY_FEATURES + PATTERN_FEATURES

FEATURE_GROUPS = {
    "composition": COMPOSITION_FEATURES,
    "entropy": ENTROPY_FEATURES,
    "pattern": PATTERN_FEATURES,
}

_BASE_WORDS = set(COMMON_RANK) | set(ENGLISH_WORDS) | set(NAMES)
_STRIP_TAIL = re.compile(r"[\d\W_]+$")


def shannon_entropy(s: str) -> float:
    """Empirical Shannon entropy H = -sum p_i log2 p_i over the characters of s (bits/char)."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _coverage(n: int, spans: list[tuple[int, int]]) -> float:
    """Fraction of positions in [0, n) covered by at least one span."""
    covered = np.zeros(n, dtype=bool)
    for i, j in spans:
        covered[i:j] = True
    return float(covered.mean()) if n else 0.0


def _longest(matches) -> int:
    return max((m.j - m.i for m in matches), default=0)


def extract_features(pw: str) -> dict:
    """Return the feature dict for a single password."""
    n = len(pw)
    classes = [char_class(c) for c in pw]
    cnt = Counter(classes)

    f = {
        # --- composition ---
        "length": n,
        "n_lower": cnt["lower"],
        "n_upper": cnt["upper"],
        "n_digit": cnt["digit"],
        "n_symbol": cnt["symbol"],
        "n_char_classes": len(cnt),
        "frac_upper": cnt["upper"] / n if n else 0.0,
        "frac_digit": cnt["digit"] / n if n else 0.0,
        "frac_symbol": cnt["symbol"] / n if n else 0.0,
    }

    # --- entropy ---
    h = shannon_entropy(pw)
    f["shannon_entropy"] = h
    f["shannon_total_bits"] = h * n
    f["pool_entropy_bits"] = n * math.log2(pool_size(pw)) if n else 0.0
    f["unique_char_ratio"] = len(set(pw)) / n if n else 0.0
    f["class_transitions"] = sum(a != b for a, b in zip(classes, classes[1:]))

    # --- patterns ---
    dmatches = dictionary_matches(pw)
    f["has_dict_word"] = int(bool(dmatches))
    f["longest_dict_word"] = _longest(dmatches)
    f["dict_coverage"] = _coverage(n, [(m.i, m.j) for m in dmatches])
    # Greedy count of non-overlapping dictionary words (longest first).
    taken = np.zeros(n, dtype=bool)
    n_words = 0
    for m in sorted(dmatches, key=lambda m: -(m.j - m.i)):
        if not taken[m.i:m.j].any():
            taken[m.i:m.j] = True
            n_words += 1
    f["n_dict_words"] = n_words
    f["has_leet"] = int(any(m.detail.endswith(":leet") for m in dmatches))
    f["longest_keyboard_run"] = _longest(spatial_matches(pw))
    f["longest_sequence"] = _longest(sequence_matches(pw))
    f["max_char_repeat"] = max((m.j - m.i for m in repeat_char_matches(pw)), default=1 if n else 0)
    f["has_block_repeat"] = int(bool(repeat_block_spans(pw)))
    dates = date_matches(pw)
    f["has_year"] = int(any(m.kind == "year" for m in dates))
    f["has_date"] = int(any(m.kind == "date" for m in dates))
    f["is_common_password"] = int(pw.lower() in COMMON_RANK or deleet(pw) in COMMON_RANK)
    base = _STRIP_TAIL.sub("", pw)
    f["common_base"] = int(len(base) >= 3 and (base.lower() in _BASE_WORDS or deleet(base) in _BASE_WORDS))
    f["word_then_digits"] = int(bool(re.fullmatch(r"[A-Za-z]{3,}\d+[\W_]?", pw)))
    f["capitalized_first_only"] = int(n > 1 and pw[0].isupper() and cnt["upper"] == 1)
    f["digits_at_end"] = int(bool(re.search(r"\d+$", pw)))
    f["symbol_at_end"] = int(n > 0 and classes[-1] == "symbol")
    return f


def build_feature_matrix(passwords) -> pd.DataFrame:
    """Feature DataFrame (columns in ``ALL_FEATURES`` order) for an iterable of passwords."""
    return pd.DataFrame([extract_features(p) for p in passwords], columns=ALL_FEATURES)
