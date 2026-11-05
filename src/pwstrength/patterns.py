"""Pattern matchers shared by the labeler and the feature extractor.

Each matcher scans a password and returns ``Match`` objects that describe a
substring an attacker could guess cheaply (a dictionary word, a keyboard walk,
an alphabetic/numeric sequence, a repeated character, a year or date). Each
match carries an estimate, in bits (log2 of guesses), of how hard that
substring is to guess *given that the attacker knows its pattern type*.

The approach follows the idea behind zxcvbn (Wheeler, 2016) in simplified form:
match patterns first, then search for the cheapest way to cover the password
(see ``labeler.estimate_bits``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .wordlists import DICTIONARIES, KEYBOARD_COLUMNS, KEYBOARD_ROWS, LEET_MAP

MIN_TOKEN = 3          # shortest substring treated as a pattern
MAX_DICT_TOKEN = 16    # longest substring looked up in dictionaries


@dataclass(frozen=True)
class Match:
    """A guessable substring ``password[i:j]``."""

    i: int          # start index (inclusive)
    j: int          # end index (exclusive)
    kind: str       # 'dictionary', 'spatial', 'sequence', 'repeat', 'date', 'year'
    bits: float     # log2(guesses) for this substring alone
    token: str      # the matched substring
    detail: str = ""  # e.g. dictionary name, 'leet', 'reversed'


# ---------------------------------------------------------------------------
# Character-class helpers
# ---------------------------------------------------------------------------
def char_class(c: str) -> str:
    """Return 'lower', 'upper', 'digit' or 'symbol' for one character."""
    if c.islower():
        return "lower"
    if c.isupper():
        return "upper"
    if c.isdigit():
        return "digit"
    return "symbol"


CLASS_POOL = {"lower": 26, "upper": 26, "digit": 10, "symbol": 33}


def pool_size(password: str) -> int:
    """Size of the brute-force alphabet implied by the classes present."""
    classes = {char_class(c) for c in password}
    return sum(CLASS_POOL[c] for c in classes) if classes else 1


def deleet(s: str) -> str:
    """Lower-case ``s`` and undo common leetspeak substitutions (p@ssw0rd -> password)."""
    return "".join(LEET_MAP.get(c, c) for c in s.lower())


# ---------------------------------------------------------------------------
# Dictionary matches
# ---------------------------------------------------------------------------
def _upper_variation_bits(token: str) -> float:
    """Extra bits for capitalisation, following zxcvbn's uppercase-variation rule."""
    letters = [c for c in token if c.isalpha()]
    n_up = sum(c.isupper() for c in letters)
    n_low = len(letters) - n_up
    if n_up == 0:
        return 0.0
    # First-letter-only, last-letter-only and ALL-CAPS are the usual habits: 1 bit.
    if (token[0].isupper() and n_up == 1) or (token[-1].isupper() and n_up == 1) or n_low == 0:
        return 1.0
    variations = sum(math.comb(n_up + n_low, k) for k in range(1, min(n_up, n_low) + 1))
    return math.log2(max(variations, 2))


def dictionary_matches(password: str) -> list[Match]:
    """All substrings found in any attacker dictionary (plain, leet or reversed)."""
    out: list[Match] = []
    n = len(password)
    lower = password.lower()
    for i in range(n):
        for j in range(i + MIN_TOKEN, min(n, i + MAX_DICT_TOKEN) + 1):
            raw = password[i:j]
            plain = lower[i:j]
            leet = deleet(raw)
            candidates = [(plain, "plain"), (leet, "leet"), (plain[::-1], "reversed")]
            for word, how in candidates:
                if how == "leet" and word == plain:
                    continue  # no substitution actually happened
                for dict_name, ranks in DICTIONARIES.items():
                    rank = ranks.get(word)
                    if rank is None:
                        continue
                    bits = math.log2(rank) + _upper_variation_bits(raw)
                    if how == "leet":
                        n_sub = sum(1 for a, b in zip(raw.lower(), word) if a != b)
                        bits += n_sub  # ~1 bit per substituted position
                    elif how == "reversed":
                        bits += 1.0
                    out.append(Match(i, j, "dictionary", bits, raw, f"{dict_name}:{how}"))
    return out


# ---------------------------------------------------------------------------
# Keyboard walks (spatial patterns)
# ---------------------------------------------------------------------------
# Map each key to (line_id, position) for every row and column it belongs to.
_KEY_LINES: dict[str, list[tuple[int, int]]] = {}
for _lid, _line in enumerate(KEYBOARD_ROWS + KEYBOARD_COLUMNS):
    for _pos, _ch in enumerate(_line):
        _KEY_LINES.setdefault(_ch, []).append((_lid, _pos))
_N_KEYS = len(_KEY_LINES)


def _adjacent_step(a: str, b: str) -> set[tuple[int, int]]:
    """Return {(line_id, direction)} for which b is the next key after a."""
    steps = set()
    for la, pa in _KEY_LINES.get(a, []):
        for lb, pb in _KEY_LINES.get(b, []):
            if la == lb and abs(pa - pb) == 1:
                steps.add((la, pb - pa))
    return steps


def spatial_matches(password: str) -> list[Match]:
    """Runs of >= 3 physically adjacent keys walking in one direction along a line."""
    out: list[Match] = []
    n = len(password)
    i = 0
    while i < n - 1:
        steps = _adjacent_step(password[i], password[i + 1])
        j = i + 1
        while steps and j < n - 1:
            nxt = steps & _adjacent_step(password[j], password[j + 1])
            if not nxt:
                break
            steps = nxt
            j += 1
        length = j - i + 1
        if steps and length >= MIN_TOKEN:
            # start key x direction x length
            bits = math.log2(_N_KEYS) + 1.0 + math.log2(length)
            out.append(Match(i, j + 1, "spatial", bits, password[i:j + 1]))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Alphabetic / numeric sequences (abc, 1234, zyx)
# ---------------------------------------------------------------------------
def sequence_matches(password: str) -> list[Match]:
    """Runs of >= 3 characters of one class with a constant step of +1 or -1."""
    out: list[Match] = []
    n = len(password)
    i = 0
    while i < n - 2:
        a, b = password[i], password[i + 1]
        delta = ord(b) - ord(a)
        if abs(delta) == 1 and char_class(a) == char_class(b) and char_class(a) != "symbol":
            j = i + 1
            while (j + 1 < n and ord(password[j + 1]) - ord(password[j]) == delta
                   and char_class(password[j + 1]) == char_class(a)):
                j += 1
            length = j - i + 1
            if length >= MIN_TOKEN:
                base = 10 if char_class(a) == "digit" else 26
                bits = math.log2(base) + 1.0 + math.log2(length)
                out.append(Match(i, j + 1, "sequence", bits, password[i:j + 1]))
                i = j
                continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# Repeats (aaaa) and repeated blocks (abcabc) - block bits are filled in by
# the labeler, which can estimate the block recursively.
# ---------------------------------------------------------------------------
def repeat_char_matches(password: str) -> list[Match]:
    """Runs of the same character, length >= 3."""
    out: list[Match] = []
    for m in re.finditer(r"(.)\1{2,}", password, flags=re.DOTALL):
        length = m.end() - m.start()
        bits = math.log2(CLASS_POOL[char_class(m.group(1))]) + math.log2(length)
        out.append(Match(m.start(), m.end(), "repeat", bits, m.group(0), "char"))
    return out


def repeat_block_spans(password: str) -> list[tuple[int, int, str, int]]:
    """Return (start, end, block, count) for blocks of >= 2 chars repeated >= 2 times."""
    spans = []
    n = len(password)
    for size in range(2, n // 2 + 1):
        for i in range(0, n - 2 * size + 1):
            block = password[i:i + size]
            count = 1
            while password[i + count * size:i + (count + 1) * size] == block:
                count += 1
            if count >= 2 and len(set(block)) > 1:
                spans.append((i, i + count * size, block, count))
    return spans


# ---------------------------------------------------------------------------
# Years and dates
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)")
_DATE_RES = [
    # ddmmyyyy / mmddyyyy with optional separators
    re.compile(r"(?<!\d)(\d{1,2})[-/._]?(\d{1,2})[-/._]?(19[5-9]\d|20[0-2]\d)(?!\d)"),
    # yyyymmdd
    re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d)[-/._]?(\d{1,2})[-/._]?(\d{1,2})(?!\d)"),
    # ddmmyy / mmddyy (six digits, two-digit year)
    re.compile(r"(?<!\d)(\d{2})[-/._]?(\d{2})[-/._]?(\d{2})(?!\d)"),
]


def _valid_day_month(a: int, b: int) -> bool:
    return (1 <= a <= 31 and 1 <= b <= 12) or (1 <= b <= 31 and 1 <= a <= 12)


def date_matches(password: str) -> list[Match]:
    """Four-digit years (1950-2029) and calendar dates in common layouts."""
    out: list[Match] = []
    for m in _YEAR_RE.finditer(password):
        out.append(Match(m.start(), m.end(), "year", math.log2(80), m.group(0)))
    for k, rx in enumerate(_DATE_RES):
        for m in rx.finditer(password):
            g = [int(x) for x in m.groups()]
            if k == 0:
                ok, years = _valid_day_month(g[0], g[1]), 80
            elif k == 1:
                ok, years = _valid_day_month(g[1], g[2]), 80
            else:
                ok, years = _valid_day_month(g[0], g[1]), 100
            if ok:
                bits = math.log2(31 * 12 * years)
                out.append(Match(m.start(), m.end(), "date", bits, m.group(0)))
    return out


def all_matches(password: str) -> list[Match]:
    """Every non-recursive match type for ``password``."""
    return (dictionary_matches(password) + spatial_matches(password)
            + sequence_matches(password) + repeat_char_matches(password)
            + date_matches(password))
