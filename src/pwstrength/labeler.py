"""Rule-based ground-truth labeler.

The label of a password comes from two parts, both documented in the README:

1. A pattern-aware guess estimate (``estimate_bits``): the password is covered
   by the cheapest sequence of pattern matches and brute-force characters.
   This approximates log2(number of guesses an informed attacker needs), in
   the spirit of zxcvbn's minimum-guesses search.
2. A small set of explicit policy rules (length floor, common-password list,
   thresholds on the estimate).

Labels: 0 = weak, 1 = medium, 2 = strong.

The ML models in this project never see ``estimate_bits``. They only see the
surface features in ``features.py``. The research question is how well cheap
features approximate this more expensive pattern-aware estimator, and which
features carry the signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from .patterns import Match, all_matches, deleet, pool_size, repeat_block_spans
from .wordlists import COMMON_RANK

LABELS = ["weak", "medium", "strong"]

# --- Policy constants (single source of truth; cited in README and paper) ---
MIN_LENGTH = 8           # shorter than this -> always weak
WEAK_MAX_BITS = 40.0     # estimate below this -> weak
STRONG_MIN_BITS = 60.0   # estimate at or above this (and length >= 10) -> strong
STRONG_MIN_LENGTH = 10
SEGMENT_PENALTY = 1.0    # bits added per pattern segment (ordering/choice cost)


@dataclass
class Estimate:
    """Result of the guess estimator for one password."""

    bits: float
    segments: list[Match] = field(default_factory=list)  # cheapest cover (patterns + brute force)


def _block_repeat_matches(password: str) -> list[Match]:
    """Matches for repeated blocks (abcabc), estimated recursively on the block."""
    out = []
    for i, j, block, count in repeat_block_spans(password):
        bits = _estimate_cached(block).bits + math.log2(count)
        out.append(Match(i, j, "repeat", bits, password[i:j], "block"))
    return out


@lru_cache(maxsize=200_000)
def _estimate_cached(password: str) -> Estimate:
    n = len(password)
    if n == 0:
        return Estimate(0.0, [])
    bf_bits = math.log2(pool_size(password))  # brute-force cost per character

    matches = all_matches(password) + _block_repeat_matches(password)
    by_end: dict[int, list[Match]] = {}
    for m in matches:
        by_end.setdefault(m.j, []).append(m)

    # Dynamic program: best[k] = min bits to cover password[:k].
    best = [0.0] + [math.inf] * n
    back: list[Match | None] = [None] * (n + 1)
    for k in range(1, n + 1):
        # Option 1: brute-force one more character.
        best[k] = best[k - 1] + bf_bits
        back[k] = Match(k - 1, k, "bruteforce", bf_bits, password[k - 1])
        # Option 2: end a pattern match here.
        for m in by_end.get(k, []):
            cost = best[m.i] + m.bits + SEGMENT_PENALTY
            if cost < best[k]:
                best[k], back[k] = cost, m

    # Recover the cheapest decomposition, merging adjacent brute-force chars.
    segs: list[Match] = []
    k = n
    while k > 0:
        m = back[k]
        segs.append(m)
        k = m.i
    segs.reverse()
    merged: list[Match] = []
    for s in segs:
        if merged and s.kind == "bruteforce" and merged[-1].kind == "bruteforce":
            p = merged[-1]
            merged[-1] = Match(p.i, s.j, "bruteforce", p.bits + s.bits, password[p.i:s.j])
        else:
            merged.append(s)

    bits = best[n]
    # A whole-password hit in the common list is cheaper still: its rank.
    for cand in (password.lower(), deleet(password)):
        if cand in COMMON_RANK:
            rank_bits = math.log2(COMMON_RANK[cand]) + (0.0 if cand == password else 1.0)
            if rank_bits < bits:
                bits = rank_bits
                merged = [Match(0, n, "dictionary", rank_bits, password, "common:whole")]
    return Estimate(bits, merged)


def estimate_bits(password: str) -> Estimate:
    """Pattern-aware estimate of log2(guesses) for ``password``."""
    return _estimate_cached(password)


def label_password(password: str) -> tuple[int, str, Estimate]:
    """Return (label_id, rule_that_fired, estimate) for one password.

    Rules are checked in order; the first that fires decides the label:
      R1 exact (or de-leeted) match in the common-password list -> weak
      R2 length < MIN_LENGTH                                     -> weak
      R3 estimate < WEAK_MAX_BITS                                -> weak
      R4 estimate >= STRONG_MIN_BITS and length >= 10            -> strong
      R5 otherwise                                               -> medium
    """
    est = estimate_bits(password)
    if password.lower() in COMMON_RANK or deleet(password) in COMMON_RANK:
        return 0, "R1_common_password", est
    if len(password) < MIN_LENGTH:
        return 0, "R2_too_short", est
    if est.bits < WEAK_MAX_BITS:
        return 0, "R3_low_guess_bits", est
    if est.bits >= STRONG_MIN_BITS and len(password) >= STRONG_MIN_LENGTH:
        return 2, "R4_high_guess_bits", est
    return 1, "R5_medium_default", est
