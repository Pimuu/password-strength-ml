#!/usr/bin/env python3
"""End-to-end reproducible pipeline.

    python run_pipeline.py            # full run (30,000 passwords)
    python run_pipeline.py --n 5000   # quicker smoke run

Steps
  1. Generate the synthetic labelled dataset   -> data/passwords.csv
  2. Extract features                          -> data/features.csv
  3. Stratified 80/20 split; 5-fold CV of three models on the training split
  4. Test-set evaluation of models and rule baselines -> results/metrics.csv, metrics.json
  5. Confusion matrices, permutation/impurity importance, LR coefficients
  6. Feature-group ablation and leave-one-family-out generalisation
  7. Weak-password pattern analysis               -> results/weakness_analysis.md
  8. Figures                                      -> results/figures/*.png
All randomness is driven by --seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pwstrength import plots  # noqa: E402
from pwstrength.features import (ALL_FEATURES, COMPOSITION_FEATURES,  # noqa: E402
                                 ENTROPY_FEATURES, FEATURE_GROUPS, PATTERN_FEATURES,
                                 build_feature_matrix)
from pwstrength.generator import build_dataset  # noqa: E402
from pwstrength.labeler import LABELS  # noqa: E402
from pwstrength.models import BASELINES, compute_metrics, confusion, make_models  # noqa: E402

MACRO_F1 = make_scorer(f1_score, average="macro")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30_000, help="number of unique passwords to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    data_dir, res_dir, fig_dir = ROOT / "data", ROOT / "results", ROOT / "results" / "figures"
    for d in (data_dir, res_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ 1-2
    log(f"Generating {args.n:,} synthetic passwords (seed={args.seed})")
    df = build_dataset(args.n, args.seed)
    df.to_csv(data_dir / "passwords.csv", index=False)
    log("Extracting features")
    X = build_feature_matrix(df.password)
    X.insert(0, "password", df.password)
    X.to_csv(data_dir / "features.csv", index=False)
    X = X[ALL_FEATURES]
    y = df.label.to_numpy()

    class_dist = df.label_name.value_counts().reindex(LABELS)
    fam_table = pd.crosstab(df.family, df.label_name).reindex(columns=LABELS, fill_value=0)
    fam_table.to_csv(res_dir / "family_by_class.csv")
    df.rule.value_counts().rename_axis("rule").reset_index(name="count").to_csv(
        res_dir / "label_rules.csv", index=False)

    # ------------------------------------------------------------------ 3
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    models = make_models(args.seed)
    rows, cms, fitted, preds = [], {}, {}, {}
    for name, model in models.items():
        log(f"5-fold CV + fit: {name}")
        cv_scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring=MACRO_F1, n_jobs=1)
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        p = model.predict(Xte)
        m = compute_metrics(yte, p)
        m.update(model=name, cv_f1_macro_mean=cv_scores.mean(), cv_f1_macro_std=cv_scores.std(),
                 fit_seconds=fit_s)
        rows.append(m)
        cms[name] = confusion(yte, p)
        fitted[name], preds[name] = model, p

    # ------------------------------------------------------------------ 4
    for name, fn in BASELINES.items():
        p = fn(Xte)
        m = compute_metrics(yte, p)
        m.update(model=name, cv_f1_macro_mean=np.nan, cv_f1_macro_std=np.nan, fit_seconds=0.0)
        rows.append(m)
        cms[name] = confusion(yte, p)
        preds[name] = p

    metrics = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
            "cv_f1_macro_mean", "cv_f1_macro_std"] + [f"{k}_{c}" for c in LABELS
                                                     for k in ("precision", "recall", "f1")] + ["fit_seconds"]
    metrics = metrics[cols]
    metrics.to_csv(res_dir / "metrics.csv", index=False, float_format="%.4f")
    for name, cm in cms.items():
        pd.DataFrame(cm, index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]).to_csv(
            res_dir / f"confusion_{name}.csv")

    learned = metrics[~metrics.model.str.startswith("Baseline")]
    best = learned.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0].model
    log(f"Best model on test macro-F1: {best}")

    # ------------------------------------------------------------------ 5
    log("Permutation importance (test set, 10 repeats) for tree models")
    imp_tables = {}
    for name in ("RandomForest", "HistGradientBoosting"):
        r = permutation_importance(fitted[name], Xte, yte, scoring=MACRO_F1, n_repeats=10,
                                   random_state=args.seed, n_jobs=-1)
        imp = pd.DataFrame({"feature": ALL_FEATURES, "importance_mean": r.importances_mean,
                            "importance_std": r.importances_std})
        group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
        imp["group"] = imp.feature.map(group_of)
        imp = imp.sort_values("importance_mean", ascending=False)
        imp.to_csv(res_dir / f"permutation_importance_{name}.csv", index=False, float_format="%.5f")
        imp_tables[name] = imp
    rf_imp = pd.DataFrame({"feature": ALL_FEATURES,
                           "impurity_importance": fitted["RandomForest"].feature_importances_}
                          ).sort_values("impurity_importance", ascending=False)
    rf_imp.to_csv(res_dir / "impurity_importance_RandomForest.csv", index=False, float_format="%.5f")
    lr = fitted["LogisticRegression"][-1]
    coef = pd.DataFrame(lr.coef_.T, index=ALL_FEATURES, columns=[f"coef_{c}" for c in LABELS])
    coef = coef.sort_values("coef_weak", ascending=False)
    coef.to_csv(res_dir / "logreg_coefficients.csv", float_format="%.4f")

    # ------------------------------------------------------------------ 6a ablation
    log("Feature-group ablation (RandomForest)")
    sets = {
        "composition only": COMPOSITION_FEATURES,
        "entropy only": ENTROPY_FEATURES,
        "composition + entropy": COMPOSITION_FEATURES + ENTROPY_FEATURES,
        "pattern only": PATTERN_FEATURES,
        "all features": ALL_FEATURES,
    }
    ab_rows = []
    for set_name, feats in sets.items():
        m = clone(models["RandomForest"])
        s = cross_val_score(m, Xtr[feats], ytr, cv=cv, scoring=MACRO_F1)
        m.fit(Xtr[feats], ytr)
        mm = compute_metrics(yte, m.predict(Xte[feats]))
        ab_rows.append({"feature_set": set_name, "n_features": len(feats), "accuracy": mm["accuracy"],
                        "f1_macro": mm["f1_macro"], "recall_weak": mm["recall_weak"],
                        "cv_f1_mean": s.mean(), "cv_f1_std": s.std()})
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(res_dir / "ablation.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------ 6b leave-one-family-out
    log("Leave-one-family-out generalisation (HistGradientBoosting)")
    lofo_rows = []
    fam = df.family.to_numpy()
    # in-distribution reference: out-of-fold predictions over the whole dataset
    oof = np.empty_like(y)
    for a, b in cv.split(X, y):
        mm = clone(models["HistGradientBoosting"]).fit(X.iloc[a], y[a])
        oof[b] = mm.predict(X.iloc[b])
    for f in sorted(df.family.unique()):
        held = fam == f
        mm = clone(models["HistGradientBoosting"]).fit(X[~held], y[~held])
        p = mm.predict(X[held])
        lofo_rows.append({
            "family": f, "n": int(held.sum()),
            "share_weak": float((y[held] == 0).mean()), "share_medium": float((y[held] == 1).mean()),
            "share_strong": float((y[held] == 2).mean()),
            "accuracy_in_dist": float((oof[held] == y[held]).mean()),
            "accuracy_heldout": float((p == y[held]).mean()),
            "f1_macro_heldout": f1_score(y[held], p, average="macro",
                                         labels=sorted(set(y[held])), zero_division=0),
        })
    lofo = pd.DataFrame(lofo_rows)
    lofo.to_csv(res_dir / "leave_one_family_out.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------ 7 weakness analysis
    log("Weak-password pattern analysis")
    flags = pd.DataFrame({
        "dictionary word (>=3 chars)": X.has_dict_word == 1,
        "dictionary covers >=50% of chars": X.dict_coverage >= 0.5,
        "common password (exact)": X.is_common_password == 1,
        "common/dictionary base + suffix": X.common_base == 1,
        "word followed by digits": X.word_then_digits == 1,
        "leetspeak substitution": X.has_leet == 1,
        "keyboard walk (>=3 keys)": X.longest_keyboard_run >= 3,
        "alpha/numeric sequence (>=3)": X.longest_sequence >= 3,
        "char repeated >=3 times in a row": X.max_char_repeat >= 3,
        "repeated block (abcabc)": X.has_block_repeat == 1,
        "contains a year (1950-2029)": X.has_year == 1,
        "contains a date": X.has_date == 1,
        "capitalised first letter only": X.capitalized_first_only == 1,
        "ends with digits": X.digits_at_end == 1,
        "shorter than 8 chars": X.length < 8,
    })
    prev = flags.groupby(df.label_name).mean().T.reindex(columns=LABELS)
    prev["weak_vs_rest_lift"] = prev["weak"] / flags[df.label != 0].mean().replace(0, np.nan)
    prev.to_csv(res_dir / "pattern_prevalence_by_class.csv", float_format="%.4f")

    # Which pattern types does the labeler's cheapest decomposition use for weak passwords?
    weak = df[df.label == 0]
    seg_share = {}
    for kind in ["dictionary", "year", "date", "spatial", "sequence", "repeat", "bruteforce"]:
        seg_share[kind] = float(weak.segments.str.split("|").apply(lambda s: kind in s).mean())
    seg_share = pd.Series(seg_share).sort_values(ascending=False)
    seg_share.rename("share_of_weak_passwords").to_csv(res_dir / "weak_decomposition_patterns.csv")

    # Where naive meters go wrong: rated strong by LUDS, labelled weak.
    luds_all = BASELINES["Baseline_LUDS_meter"](X)
    fooled = df[(luds_all == 2) & (y == 0)]
    fooled_examples = fooled.sample(min(15, len(fooled)), random_state=args.seed)[
        ["password", "family", "guess_bits", "segments"]]
    fooled_examples.to_csv(res_dir / "luds_false_strong_examples.csv", index=False)
    best_pred = preds[best]
    err = df.iloc[te][best_pred != yte].assign(predicted=[LABELS[k] for k in best_pred[best_pred != yte]])
    err[["password", "family", "label_name", "predicted", "guess_bits", "rule"]].to_csv(
        res_dir / f"errors_{best}.csv", index=False)

    # ------------------------------------------------------------------ 8 figures
    log("Rendering figures")
    plots.entropy_distributions(df.assign(**X[["shannon_entropy", "pool_entropy_bits"]]),
                                fig_dir / "entropy_distribution_by_class.png")
    plots.confusion_heatmap(cms[best], f"{best} (test set, n={len(te):,})",
                            fig_dir / "confusion_matrix_best.png")
    plots.confusion_grid({k: cms[k] for k in list(models) + list(BASELINES)},
                         fig_dir / "confusion_matrices_all.png")
    best_tree = best if best in imp_tables else "RandomForest"
    plots.feature_importance(imp_tables[best_tree], FEATURE_GROUPS, fig_dir / "feature_importance.png",
                             title=f"Permutation importance, {best_tree}")
    plots.model_comparison(metrics, fig_dir / "model_comparison.png")
    plots.ablation(ab, fig_dir / "ablation.png")
    plots.pattern_prevalence(prev[LABELS], fig_dir / "pattern_prevalence.png")
    plots.lofo(lofo, fig_dir / "leave_one_family_out.png")

    # ------------------------------------------------------------------ summary files
    summary = {
        "n_passwords": int(len(df)), "seed": args.seed, "n_train": int(len(tr)), "n_test": int(len(te)),
        "class_distribution": {k: int(v) for k, v in class_dist.items()},
        "best_model": best,
        "test_metrics": metrics.set_index("model").round(4).to_dict(orient="index"),
        "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
        "luds_false_strong_count": int(len(fooled)),
        "luds_false_strong_share_of_weak": float(len(fooled) / (y == 0).sum()),
    }
    (res_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_analysis(res_dir, df, metrics, best, imp_tables[best_tree], best_tree, rf_imp, coef, ab, lofo,
                   prev, seg_share, fooled, fooled_examples, class_dist)
    log("Done. Results in results/, figures in results/figures/")


def write_analysis(res_dir, df, metrics, best, imp, imp_model, rf_imp, coef, ab, lofo, prev, seg_share,
                   fooled, fooled_examples, class_dist):
    """Auto-generated short analysis of the most predictive weakness features."""
    m = metrics.set_index("model")
    top = imp.head(8)
    L = []
    L.append("# Weak-password pattern analysis (auto-generated by run_pipeline.py)\n")
    L.append(f"Dataset: {len(df):,} unique synthetic passwords; class counts "
             + ", ".join(f"{k} = {v:,}" for k, v in class_dist.items()) + ".\n")
    L.append("## Model performance (held-out test set)\n")
    L.append("| model | accuracy | macro-P | macro-R | macro-F1 | CV macro-F1 (mean ± std) |")
    L.append("|---|---|---|---|---|---|")
    for name, r in m.sort_values("f1_macro", ascending=False).iterrows():
        cvs = "—" if np.isnan(r.cv_f1_macro_mean) else f"{r.cv_f1_macro_mean:.4f} ± {r.cv_f1_macro_std:.4f}"
        L.append(f"| {name} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} "
                 f"| {r.f1_macro:.4f} | {cvs} |")
    L.append(f"\nBest learned model: **{best}**.\n")
    L.append(f"## Most predictive features (permutation importance, {imp_model})\n")
    L.append("| rank | feature | group | Δ macro-F1 when permuted |")
    L.append("|---|---|---|---|")
    for i, r in enumerate(top.itertuples(), 1):
        L.append(f"| {i} | `{r.feature}` | {r.group} | {r.importance_mean:.4f} ± {r.importance_std:.4f} |")
    L.append("\nTop RandomForest impurity importances: "
             + ", ".join(f"`{r.feature}` ({r.impurity_importance:.3f})" for r in rf_imp.head(6).itertuples()) + ".\n")
    L.append("Logistic-regression coefficients pushing towards **weak** (standardised features): "
             + ", ".join(f"`{f}` ({v:+.2f})" for f, v in coef.coef_weak.head(5).items())
             + "; pushing away from weak: "
             + ", ".join(f"`{f}` ({v:+.2f})" for f, v in coef.coef_weak.tail(5)[::-1].items()) + ".\n")
    L.append("## Feature-group ablation (RandomForest)\n")
    L.append("| feature set | #features | test macro-F1 | weak recall |")
    L.append("|---|---|---|---|")
    for r in ab.itertuples():
        L.append(f"| {r.feature_set} | {r.n_features} | {r.f1_macro:.4f} | {r.recall_weak:.4f} |")
    L.append("\n## Pattern prevalence by class\n")
    L.append("| pattern | weak | medium | strong | lift (weak vs rest) |")
    L.append("|---|---|---|---|---|")
    for name, r in prev.iterrows():
        lift = "∞ (absent outside weak)" if not np.isfinite(r.weak_vs_rest_lift) else f"{r.weak_vs_rest_lift:.1f}×"
        L.append(f"| {name} | {r.weak:.1%} | {r.medium:.1%} | {r.strong:.1%} | {lift} |")
    L.append("\nPattern types in the labeler's cheapest decomposition of weak passwords: "
             + ", ".join(f"{k} {v:.1%}" for k, v in seg_share.items()) + ".\n")
    L.append("## Where a composition-rule (LUDS) meter is fooled\n")
    L.append(f"{len(fooled):,} passwords ({len(fooled) / class_dist['weak']:.1%} of all weak passwords) are rated "
             "**strong** by the LUDS meter but are weak under the pattern-aware labeler. Examples:\n")
    L.append("| password | family | guess bits | decomposition |")
    L.append("|---|---|---|---|")
    for r in fooled_examples.itertuples():
        L.append(f"| `{r.password}` | {r.family} | {r.guess_bits:.1f} | {r.segments.replace('|', ' + ')} |")
    L.append("\n## Leave-one-family-out generalisation (HistGradientBoosting)\n")
    L.append("| held-out family | n | in-distribution acc. | held-out acc. |")
    L.append("|---|---|---|---|")
    for r in lofo.sort_values("accuracy_heldout").itertuples():
        L.append(f"| {r.family} | {r.n:,} | {r.accuracy_in_dist:.4f} | {r.accuracy_heldout:.4f} |")
    (res_dir / "weakness_analysis.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
