# Password-Strength Classification & Weak-Password Pattern Analysis

A small, self-contained research project. It generates a **synthetic,
rule-labelled password dataset**, extracts interpretable features, trains three
machine-learning classifiers (weak / medium / strong) and compares them with
two classic rule-based meters. It then analyses which features best predict a
weak password. A paper written from the actual results is in
[`paper/paper.md`](paper/paper.md).

> **Ethics / data note.** This project uses **no real leaked-credential data**.
> Every password is produced by `src/pwstrength/generator.py` from hand-written
> lists in `src/pwstrength/wordlists.py`, which holds 232 generic weak
> passwords, 418 English words and 90 first names, all typed in by hand.
> The ranks in those lists are illustrative orderings, not breach frequencies.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py            # full run: 30,000 passwords, about 3 minutes on 4 cores
```

```bash
python run_pipeline.py --n 5000   # quick smoke run
```

```bash
python -m unittest discover -s tests -v
```

The pipeline is deterministic for a given `--seed` (default 42). Running it
again regenerates every file in `data/` and `results/`.

## Repository layout

```
run_pipeline.py              one reproducible end-to-end pipeline
requirements.txt             numpy, pandas, scikit-learn, matplotlib
src/pwstrength/
  wordlists.py               hand-written dictionaries, keyboard layout, leet map
  patterns.py                pattern matchers (dictionary/leet/reversed, keyboard walk,
                             sequence, repeat, year/date) with per-match guess bits
  labeler.py                 zxcvbn-style minimum-guess estimator + labelling rules
  generator.py               10 password "families" -> synthetic labelled dataset
  features.py                31 per-password features in 3 groups
  models.py                  LogisticRegression / RandomForest / HistGradientBoosting,
                             LUDS & pool-entropy baselines, metric helpers
  plots.py                   all figures
tests/test_pipeline.py       unit tests (labeler, matchers, features, generator)
data/                        generated dataset + feature matrix (CSV)
results/                     metrics, confusion matrices, importances, analyses
results/figures/             PNG figures used in the paper
paper/paper.md               the paper
```

## How the dataset is generated

`build_dataset(n=30000, seed=42)` draws passwords from ten **families**. Each
family imitates a password-creation habit:

| family | example shape | weight |
|---|---|---|
| `common` | a hand-listed weak password, sometimes capitalised or suffixed (`Monkey123`) | 0.05 |
| `word_digits` | word/name + digits or year, optional symbol (`tiger1987!`) | 0.08 |
| `leet` | leetspeak word + optional suffix (`Dr4g0n99`) | 0.07 |
| `keyboard` | concatenated keyboard walks (`1qaz2wsx`, `asdfgh!`) | 0.05 |
| `name_date` | name + date or year (`Anna_12.05.1994`) | 0.05 |
| `word_combo` | two words with a separator (`Coffee.Ethan12`) | 0.08 |
| `passphrase` | 3–6 uniformly chosen words (`maple-orbit-cello-ring`) | 0.15 |
| `mangled` | "complex-looking" human password: leet word + symbol + word + digits (+ random chunk) | 0.17 |
| `pattern` | repeats and sequences (`aaaaaa`, `abcabc`, `abcd1234`) | 0.03 |
| `random` | uniformly random string, 6–20 chars, random subset of classes | 0.27 |

Duplicates are removed, so no password can appear in both the train and test
splits. The family is kept as metadata for the leave-one-family-out
experiment. It is **never** a model feature.

## How labels are derived (the rules)

`labeler.label_password` applies these rules in order. The first rule that
fires decides the label:

| rule | condition | label |
|---|---|---|
| R1 | password (lower-cased or de-leeted) is in the common-password list | weak |
| R2 | length < 8 | weak |
| R3 | pattern-aware guess estimate < 40 bits | weak |
| R4 | estimate ≥ 60 bits **and** length ≥ 10 | strong |
| R5 | otherwise | medium |

The **pattern-aware guess estimate** is a simplified version of zxcvbn's
minimum-guess search (Wheeler, 2016):

1. Find every guessable substring: dictionary words (plain, leetspeak or
   reversed, with a capitalisation penalty), keyboard walks along QWERTY rows
   and columns, ±1 character sequences, repeated characters or blocks, and
   years or dates. Each match costs `log2(guesses)` bits. For a dictionary
   word that is `log2(rank)` plus variation bits.
2. Cover the whole password with the cheapest mix of matches and brute-force
   characters, using dynamic programming. A brute-force character costs
   `log2(pool)`, where `pool` is the alphabet size implied by the character
   classes present (26/26/10/33). Each pattern segment adds 1 extra bit.

So the label depends on **length, character classes (through the brute-force
pool), dictionary words, common patterns and entropy**, as the task requires.
40 bits ≈ 1.1×10¹² guesses and 60 bits ≈ 1.2×10¹⁸ guesses. These values
bracket the ~10¹⁴-guess offline-attack threshold of Florêncio, Herley & van
Oorschot (2014). They are design parameters, defined as constants in
`labeler.py`.

## Features (31, in three groups)

* **Composition (9):** `length`, `n_lower`, `n_upper`, `n_digit`, `n_symbol`,
  `n_char_classes`, `frac_upper`, `frac_digit`, `frac_symbol`.
* **Entropy (5):** `shannon_entropy` (bits/char), `shannon_total_bits`,
  `pool_entropy_bits` (length × log2 pool), `unique_char_ratio`,
  `class_transitions`.
* **Pattern (17):** `has_dict_word`, `longest_dict_word`, `dict_coverage`,
  `n_dict_words`, `has_leet`, `longest_keyboard_run`, `longest_sequence`,
  `max_char_repeat`, `has_block_repeat`, `has_year`, `has_date`,
  `is_common_password`, `common_base`, `word_then_digits`,
  `capitalized_first_only`, `digits_at_end`, `symbol_at_end`.

## Design choices (and why)

* **Synthetic data with a transparent oracle.** Real leaks are ethically and
  legally fraught, and the task forbids them. A rule-based oracle makes the
  label *auditable*: every label comes with the rule that fired
  (`data/passwords.csv`, columns `rule` and `segments`).
* **Learn from cheap features, not from the oracle.** The guess estimate
  (`guess_bits`) is left out of the features. Otherwise the task would be
  trivial (label leakage). The research question is how well cheap,
  interpretable features approximate a pattern-aware estimator, and which
  ones matter most.
* **Three model types.** Logistic regression is the linear, interpretable
  reference. RandomForest and HistGradientBoosting are non-linear tree
  ensembles suited to mixed count and flag features. All use balanced class
  weights because *medium* is the smallest class.
* **Two non-learned baselines.** A LUDS composition meter (the "8 chars +
  upper + digit + symbol" style) and a pool-entropy meter (length × log2
  alphabet). These are what many websites still deploy, and they show what
  the pattern features add.
* **Macro-F1 as the main metric.** Classes are imbalanced (weak ≈ 51%), so
  accuracy alone would flatter a weak-biased model.
* **Permutation importance on the test set** (10 repeats) is the main
  importance measure. Impurity importance favours high-cardinality continuous
  features. It is reported too, for comparison.
* **Ablation and leave-one-family-out (LOFO).** The ablation measures what
  each feature group adds. LOFO tests whether a model generalises to a
  password-creation habit it never saw during training. This is the main
  external-validity risk of synthetic data.
* **Reproducibility.** One seed drives generation, splits, models and
  permutations. Duplicates are removed before splitting.

## Results

Numbers below come from the full run (30,000 passwords, seed 42, 24,000 train / 6,000 test).
Running the pipeline twice reproduces them exactly.

| model | accuracy | macro-F1 | 5-fold CV macro-F1 |
|---|---|---|---|
| **HistGradientBoosting** | **0.9647** | **0.9545** | 0.9533 ± 0.0023 |
| RandomForest | 0.9573 | 0.9435 | 0.9461 ± 0.0028 |
| LogisticRegression | 0.9348 | 0.9194 | 0.9209 ± 0.0021 |
| Baseline: pool entropy | 0.4768 | 0.4282 | — |
| Baseline: LUDS meter | 0.4505 | 0.4167 | — |

Key findings:

* **Composition and entropy meters fail on human passwords.** 34.2% of all
  weak passwords (5,230) are rated *strong* by the LUDS meter, for example
  `TigerPeter1973` and `peach#Emily2019`.
* **The most predictive weakness features** (test-set permutation importance,
  Δ macro-F1) are `dict_coverage` (0.335), `pool_entropy_bits` (0.324),
  `has_year` (0.095), `common_base` (0.062) and `longest_keyboard_run`
  (0.044). `n_char_classes` contributes almost nothing (0.002).
* **Pattern features are essential.** RandomForest macro-F1 rises from
  0.847 (composition + entropy) to 0.944 (all features).
* **The weakest signature** is "known word + predictable suffix": 39.2% of
  weak passwords versus 0.7% of medium and 0% of strong.
* **Unseen habits break learned meters.** With a family held out of
  training, accuracy drops to 0.343 (keyboard walks), 0.392 (passphrases)
  and 0.516 (name+date).

Outputs in `results/`:

| file | content |
|---|---|
| `metrics.csv`, `metrics.json` | accuracy, macro/weighted and per-class precision/recall/F1, CV scores, confusion matrices |
| `confusion_<model>.csv` | confusion matrix per model and baseline |
| `permutation_importance_*.csv`, `impurity_importance_RandomForest.csv`, `logreg_coefficients.csv` | feature importance |
| `ablation.csv`, `leave_one_family_out.csv` | extra experiments |
| `pattern_prevalence_by_class.csv`, `weak_decomposition_patterns.csv`, `family_by_class.csv`, `label_rules.csv` | dataset / weakness analysis |
| `luds_false_strong_examples.csv`, `errors_HistGradientBoosting.csv` | qualitative examples |
| `weakness_analysis.md` | auto-generated short analysis of the most predictive weakness features |
| `figures/feature_importance.png` | permutation importance, coloured by feature group |
| `figures/entropy_distribution_by_class.png` | Shannon / pool / pattern-aware entropy by class |
| `figures/confusion_matrix_best.png`, `figures/confusion_matrices_all.png` | confusion-matrix heatmaps |
| `figures/model_comparison.png`, `figures/ablation.png`, `figures/pattern_prevalence.png`, `figures/leave_one_family_out.png` | further figures used in the paper |


## Limitations (short)

The labels come from our own estimator, so the models learn to imitate *that*
estimator, not real attacker behaviour. The dictionaries are small (740 entries), so passphrases built from them are rated weaker than they would be
with a 7,776-word Diceware list. See paper §5 for the full discussion.

## License

MIT. The code and the hand-written word lists may be reused freely.
