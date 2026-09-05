#!/usr/bin/env python3
"""Fit the hazard fusion weights on DEV -- the step build-plan.py D11 specified and that
``plumerank/constants.py`` has been standing in for with a-priori guesses.

WHAT THIS CLOSES
----------------
``build-plan.md`` D11 says ``rank.py`` fuses S1-S4 with "a weighted logistic, weights fit on
DEV". They were never fit. ``constants.py`` carries FUSION_BIAS/W_S1..W_S4/W_ESCALATION as
values written before any data was seen, and every number the project has published so far --
including the CONFUSE per-class split that came within a hair of aborting the project on
2026-09-05 -- was produced by that unfitted instrument. ``results/gonogo_2026-09-01.md`` §7.3
names this as one of two unmet preconditions.

THE METHOD, FIXED BEFORE THE FIT
--------------------------------
* **L2-regularised logistic regression by Newton-Raphson (IRLS)**, numpy only. No new
  dependency: ``requirements.lock`` is a judged artifact and a fitted constant is not worth
  a line in it.
* **Raw feature units, no standardisation.** The UI shows each row's S1/S2/S3 chips beside
  the hazard, so a weight has to mean something in the units the chip displays. A
  standardised weight would be uninterpretable at exactly the place the design promises
  interpretability.
* **Balanced class weights.** Post-ignition frames are the minority; unweighted, the fit
  would buy its likelihood by predicting "no fire" everywhere, which is the one answer the
  system may not give.
* **The bias is not penalised**, so it stays a free parameter of the data rather than an
  artefact of LAMBDA. Note what balanced weighting then implies: the fitted bias sits near
  ZERO by construction, because reweighting makes the two classes equal mass. **The bias
  therefore does not carry the base rate** -- the operating point is set downstream by
  ``rank.solve_threshold`` bisecting on CONFUSE, and that separation is deliberate. A bias
  that encoded DEV's base rate would silently fight the budget solver for control of the
  same quantity.
* **LAMBDA is fixed a priori, not tuned.** There is no second held-out set inside DEV to tune
  it on, and tuning it against the grouped CV below would make that CV an in-sample number.

WHAT IS REPORTED, AND WHY THE GROUPED CV IS THE ONLY HONEST ACCURACY
--------------------------------------------------------------------
Frames inside one sequence are the same fire, the same camera and the same weather, thirty
seconds apart. An in-sample AUC over frames therefore measures how well the model memorised
~30 sequences, not how well it will rank the 31st. **Leave-one-sequence-out** is the number
that means something, and both are printed so the gap is visible rather than implied.
"""

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys

import numpy as np

FEATURES = ("s1", "s2", "s3", "s4", "esc")

#: L2 strength on the weights (never the bias). Fixed a priori -- see the module docstring.
LAMBDA = 1.0

#: Newton-Raphson terminates on either of these. Both are convergence tolerances, not
#: modelling choices: any value small enough gives the same fit.
MAX_ITERS = 200
TOL = 1e-10


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty -- run bench.py --arm dev --dump-features first")
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    groups = np.array([r["sequence_id"] for r in rows])
    return X, y, groups, rows


def _sigmoid(z):
    # Branch-free and overflow-free: exp(-|z|) never overflows, and the two branches are
    # algebraically identical.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit(X, y, lam=LAMBDA):
    """L2 logistic by IRLS with balanced class weights. Returns (bias, weights)."""
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])          # design matrix, bias in column 0
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            f"DEV features carry only one class (pos={int(n_pos)} neg={int(n_neg)}). "
            "A logistic cannot be fit on it and a constant hazard is not a model."
        )
    sw = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))

    # Penalty matrix: zero on the bias, lam on every weight.
    P = np.eye(d + 1) * lam
    P[0, 0] = 0.0

    beta = np.zeros(d + 1)
    for _ in range(MAX_ITERS):
        p = _sigmoid(A @ beta)
        W = sw * p * (1.0 - p)
        grad = A.T @ (sw * (y - p)) - P @ beta
        H = (A.T * W) @ A + P
        # Ridge on the Hessian only if it is singular -- a feature that is constant across
        # the whole arm (S4 with the ablation off, say) makes it so, and the fit should
        # still return rather than raise.
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-8 * np.eye(d + 1), grad)
        beta = beta + step
        if np.max(np.abs(step)) < TOL:
            break
    return float(beta[0]), beta[1:].astype(float)


def auc(scores, y):
    """Mann-Whitney U / (n1*n2), ties at half.

    The same ESTIMATOR as ``spike/s3_separation.py:auc``, so the AUCs the two documents
    publish are comparable -- but a different CALLING CONVENTION: the gate passes two
    already-split class arrays, this takes scores alongside their labels, because here the
    split is by ``y`` and doing it at the call site would duplicate it at every caller.
    """
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def grouped_cv(X, y, groups):
    """Leave-one-SEQUENCE-out. Held-out scores are pooled and scored once, rather than
    averaged per fold: most folds are a single sequence and therefore a single class, where
    a per-fold AUC is undefined."""
    held = np.zeros(len(y))
    seqs = sorted(set(groups.tolist()))
    for s in seqs:
        te = groups == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            held[te] = np.nan
            continue
        b, w = fit(X[tr], y[tr])
        held[te] = b + X[te] @ w
    ok = ~np.isnan(held)
    return auc(held[ok], y[ok]), int(ok.sum()), len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=pathlib.Path,
                    default=pathlib.Path("results/features_dev.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    X, y, groups, rows = load(args.features)
    n_comp = int(sum(r["has_component"] for r in rows))
    per_seq = collections.Counter(groups.tolist())

    print(f"{len(rows)} frames · {len(per_seq)} sequences · {int(y.sum())} positive · "
          f"{len(rows) - int(y.sum())} negative · {n_comp} carry a component\n")

    bias, w = fit(X, y)
    in_sample = auc(bias + X @ w, y)
    cv_auc, n_scored, n_folds = grouped_cv(X, y, groups)

    print("fitted weights (raw units, as the UI chips display them):")
    print(f"  FUSION_BIAS          {bias:+.4f}")
    for name, val in zip(FEATURES, w):
        key = "FUSION_W_ESCALATION" if name == "esc" else f"FUSION_W_{name.upper()}"
        print(f"  {key:<20} {val:+.4f}")

    print(f"\nAUC in-sample (frames)          {in_sample:.4f}   <- memorisation, not accuracy")
    print(f"AUC leave-one-sequence-out      {cv_auc:.4f}   <- the honest number "
          f"({n_folds} folds, {n_scored} frames scored)")

    negative = [k for k, v in zip(FEATURES, w) if v < 0]
    if negative:
        print(f"\nFINDING: {', '.join(negative)} fitted NEGATIVE. The spec asserts every "
              f"statistic is\nevidence FOR fire, so a negative weight means the data "
              f"contradict the premise for that\nstatistic on DEV. Publish it; do not clip "
              f"it to zero.")

    payload = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "features_file": str(args.features),
        "method": "L2 logistic (IRLS), balanced class weights, bias unpenalised",
        "lambda": LAMBDA,
        "n_frames": len(rows),
        "n_sequences": len(per_seq),
        "n_positive": int(y.sum()),
        "n_with_component": n_comp,
        "weights": {"FUSION_BIAS": bias,
                    **{("FUSION_W_ESCALATION" if n == "esc" else f"FUSION_W_{n.upper()}"): float(v)
                       for n, v in zip(FEATURES, w)}},
        "auc_in_sample_frames": in_sample,
        "auc_leave_one_sequence_out": cv_auc,
        "cv_folds": n_folds,
        "negative_weights": negative,
    }
    out = args.out or pathlib.Path("results") / f"fusion_fit_{dt.date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
