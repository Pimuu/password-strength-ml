"""Synthetic password generator.

Passwords come from ten hand-designed "families". Each family imitates a
password-creation habit described in the usability literature (word plus
digits, leetspeak, keyboard walks, dates, passphrases, etc.), and one family
is uniformly random strings. Every password is built from the hand-written
lists in ``wordlists.py``. No real credential data is used.

The family of each password is kept as metadata, for the leave-one-family-out
generalisation experiment. It is never given to the models as a feature.
"""

from __future__ import annotations

import random
import string

import pandas as pd

from .labeler import LABELS, label_password
from .wordlists import (COMMON_PASSWORDS, ENGLISH_WORDS, KEYBOARD_COLUMNS,
                        KEYBOARD_ROWS, LEET_FORWARD, NAMES, SYMBOLS)

# Relative sampling weight of each family (before de-duplication).
FAMILY_WEIGHTS = {
    "common": 0.05,
    "word_digits": 0.08,
    "leet": 0.07,
    "keyboard": 0.05,
    "name_date": 0.05,
    "word_combo": 0.08,
    "passphrase": 0.15,
    "mangled": 0.17,
    "pattern": 0.03,
    "random": 0.27,
}

COMMON_SUFFIXES = ["1", "12", "123", "1234", "!", "!!", "1!", "01", "007", "69", "99", "?"]
SEPARATORS = ["", "-", "_", ".", " ", "@", "#", "!", "+"]


class PasswordGenerator:
    """Draw synthetic passwords from the families above, reproducibly."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ---------------- small helpers ----------------
    def _word(self, min_len: int = 3) -> str:
        pool = [w for w in ENGLISH_WORDS if len(w) >= min_len]
        # Skew towards frequent words: half the time pick from the first quarter.
        if self.rng.random() < 0.5:
            pool = pool[: max(1, len(pool) // 4)]
        return self.rng.choice(pool)

    def _year(self) -> str:
        return str(self.rng.randint(1950, 2029))

    def _digits(self, lo: int = 1, hi: int = 4) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(self.rng.randint(lo, hi)))

    def _case(self, w: str) -> str:
        r = self.rng.random()
        if r < 0.55:
            return w
        if r < 0.85:
            return w.capitalize()
        if r < 0.93:
            return w.upper()
        return "".join(c.upper() if self.rng.random() < 0.5 else c for c in w)

    def _leetify(self, w: str, p: float = 0.6) -> str:
        return "".join(self.rng.choice(LEET_FORWARD[c]) if c in LEET_FORWARD and self.rng.random() < p
                       else c for c in w)

    # ---------------- families ----------------
    def common(self) -> str:
        pw = self.rng.choice(COMMON_PASSWORDS)
        if self.rng.random() < 0.5:
            pw = self._case(pw)
            if self.rng.random() < 0.6:
                pw += self.rng.choice(COMMON_SUFFIXES + [self._year()])
        return pw

    def word_digits(self) -> str:
        base = self.rng.choice([self._word(), self.rng.choice(NAMES)])
        tail = self._year() if self.rng.random() < 0.4 else self._digits(1, 4)
        pw = self._case(base) + tail
        if self.rng.random() < 0.35:
            pw += self.rng.choice("!@#$*?.")
        return pw

    def leet(self) -> str:
        pw = self._case(self._leetify(self._word(min_len=5)))
        if self.rng.random() < 0.6:
            pw += self.rng.choice(COMMON_SUFFIXES + [self._digits(2, 4), self._year()])
        return pw

    def keyboard(self) -> str:
        def walk() -> str:
            line = self.rng.choice(KEYBOARD_ROWS + KEYBOARD_COLUMNS * 2)
            k = self.rng.randint(3, min(10, len(line)))
            start = self.rng.randint(0, len(line) - k)
            seg = line[start:start + k]
            return seg[::-1] if self.rng.random() < 0.2 else seg
        pw = walk()
        while len(pw) < 6 or self.rng.random() < 0.35:
            pw += walk()
        if self.rng.random() < 0.4:
            pw += self.rng.choice(COMMON_SUFFIXES)
        return pw[:24]

    def name_date(self) -> str:
        name = self._case(self.rng.choice(NAMES))
        d, m = self.rng.randint(1, 28), self.rng.randint(1, 12)
        y = self._year()
        sep = self.rng.choice(["", "", ".", "-", "/"])
        style = self.rng.random()
        if style < 0.35:
            date = f"{d:02d}{sep}{m:02d}{sep}{y}"
        elif style < 0.55:
            date = f"{m:02d}{sep}{d:02d}{sep}{y[2:]}"
        else:
            date = y
        pw = name + self.rng.choice(["", "", "_", "@"]) + date
        if self.rng.random() < 0.3:
            pw += self.rng.choice("!*#")
        return pw

    def word_combo(self) -> str:
        a, b = self._case(self._word()), self._case(self.rng.choice([self._word(), self.rng.choice(NAMES)]))
        pw = a + self.rng.choice(SEPARATORS) + b
        if self.rng.random() < 0.5:
            pw += self.rng.choice([self._digits(1, 3), self._year()])
        return pw

    def passphrase(self) -> str:
        k = self.rng.randint(3, 6)
        words = [self.rng.choice(ENGLISH_WORDS) for _ in range(k)]  # uniform choice
        if self.rng.random() < 0.4:
            words = [w.capitalize() for w in words]
        pw = self.rng.choice(["-", " ", ".", "_", "", "+"]).join(words)
        if self.rng.random() < 0.4:
            pw += self._digits(1, 3)
        return pw

    def mangled(self) -> str:
        """Human 'complex-looking' passwords that pass composition policies."""
        parts = [self._case(self._leetify(self._word(), 0.3))]
        parts.append(self.rng.choice("!@#$%&*?"))
        parts.append(self._case(self.rng.choice([self._word(), self.rng.choice(NAMES)])))
        parts.append(self.rng.choice([self._digits(1, 3), self._year()]))
        if self.rng.random() < 0.4:
            parts.append(self.rng.choice("!@#$%&*?"))
        if self.rng.random() < 0.45:  # a short "random-looking" chunk some users add
            alphabet = string.ascii_letters + string.digits + "!@#$%&*?"
            parts.append("".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(2, 5))))
        if self.rng.random() < 0.3:
            self.rng.shuffle(parts)
        return "".join(parts)

    def pattern(self) -> str:
        r = self.rng.random()
        if r < 0.25:   # single repeated character
            return self.rng.choice(string.ascii_letters + string.digits + "!*") * self.rng.randint(4, 14)
        if r < 0.5:    # repeated block
            block = "".join(self.rng.choice(string.ascii_lowercase + string.digits)
                            for _ in range(self.rng.randint(2, 4)))
            return block * self.rng.randint(2, 4)
        if r < 0.75:   # sequences
            alpha = self.rng.choice([string.ascii_lowercase, string.digits, string.ascii_uppercase])
            k = self.rng.randint(4, min(12, len(alpha)))
            s = self.rng.randint(0, len(alpha) - k)
            seq = alpha[s:s + k]
            return seq[::-1] if self.rng.random() < 0.3 else seq
        # sequence + sequence, e.g. abcd1234
        k = self.rng.randint(3, 6)
        return string.ascii_lowercase[:k] + string.digits[1:k + 1]

    def random(self) -> str:
        n = self.rng.randint(6, 20)
        pools = [string.ascii_lowercase, string.ascii_uppercase, string.digits, SYMBOLS]
        chosen = [p for p in pools if self.rng.random() < 0.6] or [string.ascii_lowercase]
        alphabet = "".join(chosen)
        return "".join(self.rng.choice(alphabet) for _ in range(n))

    # ---------------- dataset ----------------
    def sample(self) -> tuple[str, str]:
        fams, weights = zip(*FAMILY_WEIGHTS.items())
        fam = self.rng.choices(fams, weights=weights, k=1)[0]
        return getattr(self, fam)(), fam


def build_dataset(n: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` unique labelled passwords.

    Duplicates are dropped, so a password can never land in both the training
    and test split. Each row records the family, the label, the rule that
    fired, and the estimator's bit score (bit score kept for analysis only).
    """
    gen = PasswordGenerator(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        pw, fam = gen.sample()
        if not pw or pw in seen:
            continue
        seen.add(pw)
        label, rule, est = label_password(pw)
        rows.append({
            "password": pw,
            "family": fam,
            "label": label,
            "label_name": LABELS[label],
            "rule": rule,
            "guess_bits": round(est.bits, 3),
            "segments": "|".join(s.kind for s in est.segments),
        })
    return pd.DataFrame(rows)
