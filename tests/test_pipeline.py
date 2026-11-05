"""Sanity tests for the labeler, matchers and feature extractor.

Run with:  python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pwstrength.features import ALL_FEATURES, extract_features, shannon_entropy  # noqa: E402
from pwstrength.generator import build_dataset  # noqa: E402
from pwstrength.labeler import label_password  # noqa: E402
from pwstrength.patterns import deleet, sequence_matches, spatial_matches  # noqa: E402


class TestLabeler(unittest.TestCase):
    def test_common_passwords_are_weak(self):
        for pw in ["password", "123456", "qwerty", "P@ssw0rd", "letmein"]:
            self.assertEqual(label_password(pw)[0], 0, pw)

    def test_short_is_weak(self):
        self.assertEqual(label_password("xQ9#a")[1], "R2_too_short")

    def test_composition_compliant_but_patterned_is_weak(self):
        # Satisfies LUDS rules (upper, lower, digit, symbol, length >= 8) but is word + year.
        self.assertEqual(label_password("Sunshine2023!")[0], 0)

    def test_long_random_is_strong(self):
        self.assertEqual(label_password("xK9#mQ2$vL7pW4&z")[0], 2)

    def test_repeated_block_is_cheap(self):
        bits = [label_password("aZ3$" * k)[2].bits for k in (1, 2)]
        # Repeated block should be cheap: second is far below 2x the first.
        self.assertLess(bits[1], 2 * bits[0])


class TestPatterns(unittest.TestCase):
    def test_deleet(self):
        self.assertEqual(deleet("P@55w0rd"), "password")

    def test_keyboard_walk(self):
        self.assertEqual(max(m.j - m.i for m in spatial_matches("xxqwertyxx")), 6)

    def test_sequence(self):
        self.assertEqual(max(m.j - m.i for m in sequence_matches("ab12345cd")), 5)


class TestFeatures(unittest.TestCase):
    def test_shannon(self):
        self.assertAlmostEqual(shannon_entropy("abcd"), 2.0)
        self.assertEqual(shannon_entropy("aaaa"), 0.0)

    def test_feature_keys_and_values(self):
        f = extract_features("Michael1987!")
        self.assertEqual(list(f), ALL_FEATURES)
        self.assertEqual(f["length"], 12)
        self.assertEqual(f["n_char_classes"], 4)
        self.assertEqual(f["has_year"], 1)
        self.assertEqual(f["capitalized_first_only"], 1)
        self.assertTrue(math.isclose(f["pool_entropy_bits"], 12 * math.log2(95)))


class TestGenerator(unittest.TestCase):
    def test_reproducible_and_unique(self):
        a, b = build_dataset(500, seed=7), build_dataset(500, seed=7)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.password.nunique(), len(a))
        self.assertEqual(set(a.label.unique()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
