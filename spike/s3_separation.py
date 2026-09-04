#!/usr/bin/env python3
"""S3 separability gate — does diffuse-boundary contrast separate smoke from fog?

The GO/NO-GO instrument for 2026-09-01. It answers ONE question and nothing else:
is S3's AUC compatible with a usable discriminator?

It deliberately does NOT test stabilisation, MOG2, tracking, the hazard fusion or the
budget solver. A rate per camera-night is not computable from static crops; that
question moves to the D11 budget checkpoint.

Usage:
    python3 s3_separation.py                    # v0, the raw-frame statistic
    python3 s3_separation.py --variant v5       # temporal-difference rescue
    python3 s3_separation.py --all-variants     # v0..v5, for the D3 selection pass

Decision rule (95% bootstrap CI on AUC), fixed before any number exists:
    CI upper  < 0.70  -> NO-GO          switch to FlashBudget
    CI lower >= 0.70  -> GO             proceed
    straddles   0.70  -> CONDITIONAL GO proceed; D11 becomes a hard abort gate
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FRAMES = HERE / "frames"
BOXES = HERE / "boxes.json"
EPS = 1e-8
BOOTSTRAP_DRAWS = 2000
CUTLINE = 0.70
SEED = 20260901  # fixed: the gate must give the same answer twice


# --- the statistic -------------------------------------------------------------

def s3(roi, variant="v0"):
    """Interior blur energy / boundary edge energy.

    Smoke's edge is soft, so interior Laplacian energy is high relative to the
    Sobel energy in the boundary ring. A cloud shadow or fog bank with a hard
    edge inverts that ratio.
    """
    roi = roi.astype(np.float32)
    h, w = roi.shape
    b = max(3, int(0.15 * min(w, h)))  # band width, fixed a priori — never fitted
    if variant == "v2b":
        b = max(3, b // 2)

    # Interior/ring are built geometrically, NOT by cv2.erode: erode's default
    # borderValue is +inf, so eroding a solid mask leaves it solid, the ring comes
    # back empty and every score is NaN.
    if h - 2 * b < 2 or w - 2 * b < 2:
        return np.nan                  # box too small to hold a boundary band
    interior = np.zeros((h, w), bool)
    interior[b:h - b, b:w - b] = True
    ring = ~interior

    if variant == "v1":  # multi-scale: take the strongest responding scale
        lap = np.max([np.abs(cv2.Laplacian(cv2.GaussianBlur(roi, (0, 0), s), cv2.CV_32F))
                      for s in (1.0, 2.0, 4.0)], axis=0)
    else:
        lap = np.abs(cv2.Laplacian(roi, cv2.CV_32F))

    gx = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))

    num = float(lap[interior].mean())
    den = float((gx + gy)[ring].mean()) + EPS
    val = num / den

    if variant == "v3":
        val = float(np.log(val + EPS))
    if variant == "v4":
        val = val / (float(roi.mean()) + EPS)
    return val


def score_box(rec, variant):
    """One scalar per box. v5 measures the statistic on a temporal difference frame."""
    img = cv2.imread(str(FRAMES / rec["file"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"missing frame: {rec['file']}")
    x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]

    if variant == "v5":
        prev_name = rec.get("prev")
        if not prev_name:
            return np.nan  # no earlier frame for this box; excluded, counted, reported
        prev = cv2.imread(str(FRAMES / prev_name), cv2.IMREAD_GRAYSCALE)
        if prev is None or prev.shape != img.shape:
            return np.nan
        img = cv2.absdiff(img, prev)

    return s3(img[y:y + h, x:x + w], variant)


# --- decision ------------------------------------------------------------------

def auc(pos, neg):
    """Mann-Whitney U / (n1*n2). Ties counted as half, as they must be."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks within ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def bootstrap_ci(pos, neg, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Resample each class independently, with replacement. The interval is the
    point of this gate — 120 boxes cannot resolve a 0.65/0.75 band on a point
    estimate, and a coin flip wearing a number is worse than no test."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = [auc(rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True)) for _ in range(draws)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def verdict(lo, hi):
    if hi < CUTLINE:
        return "NO-GO", "The data rule out even a mediocre discriminator. Switch to FlashBudget."
    if lo >= CUTLINE:
        return "GO", "Proceed. The D11 checkpoint is confirmation, not rescue."
    return "CONDITIONAL GO", "Proceed, and the D11 budget checkpoint becomes a hard abort gate."


def histogram(pos, neg, bins=20):
    lo, hi = min(min(pos), min(neg)), max(max(pos), max(neg))
    edges = np.linspace(lo, hi, bins + 1)
    ph, _ = np.histogram(pos, edges)
    nh, _ = np.histogram(neg, edges)
    out = ["", f"  {'bin':>10}  smoke                fog", ""]
    for i in range(bins):
        out.append(f"  {edges[i]:10.3f}  {'#' * int(ph[i]):<18} {'.' * int(nh[i])}")
    return "\n".join(out)


def run(variant, boxes):
    scored = [(r["class"], score_box(r, variant)) for r in boxes]
    dropped = sum(1 for _, v in scored if not np.isfinite(v))
    smoke = [v for c, v in scored if c == "smoke" and np.isfinite(v)]
    fog = [v for c, v in scored if c == "fog" and np.isfinite(v)]
    if len(smoke) < 10 or len(fog) < 10:
        raise SystemExit(f"{variant}: too few usable boxes (smoke={len(smoke)} fog={len(fog)})")
    a = auc(smoke, fog)
    lo, hi = bootstrap_ci(smoke, fog)
    overlap = sum(1 for s in smoke if s <= max(fog)) + sum(1 for f in fog if f >= min(smoke))
    return dict(variant=variant, auc=a, lo=lo, hi=hi, n_smoke=len(smoke), n_fog=len(fog),
                dropped=dropped, med_smoke=float(np.median(smoke)), med_fog=float(np.median(fog)),
                overlap=overlap, smoke=smoke, fog=fog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0",
                    choices=["v0", "v1", "v2b", "v3", "v4", "v5"])
    ap.add_argument("--all-variants", action="store_true",
                    help="D3 selection pass — discloses the multiple comparison")
    args = ap.parse_args()

    boxes = json.loads(BOXES.read_text())
    print(f"OpenCV {cv2.__version__} · {len(boxes)} boxes · cutline {CUTLINE} · "
          f"{BOOTSTRAP_DRAWS} bootstrap draws · seed {SEED}\n")

    variants = ["v0", "v1", "v2b", "v3", "v4", "v5"] if args.all_variants else [args.variant]
    results = []
    for v in variants:
        try:
            results.append(run(v, boxes))
        except SystemExit as e:
            print(f"  {e}", file=sys.stderr)

    if not results:
        raise SystemExit(
            "No variant produced a usable score. Every box was dropped — check that\n"
            "boxes.json holds boxes with sensible w/h and that spike/frames/ is populated.")

    for r in results:
        v, lab = verdict(r["lo"], r["hi"])
        print(f"{r['variant']:>4}  AUC {r['auc']:.3f}  95% CI [{r['lo']:.3f}, {r['hi']:.3f}]  "
              f"n={r['n_smoke']}/{r['n_fog']}  dropped={r['dropped']}  "
              f"med {r['med_smoke']:.3f} vs {r['med_fog']:.3f}  overlap={r['overlap']}  -> {v}")

    best = max(results, key=lambda r: r["auc"])
    v, lab = verdict(best["lo"], best["hi"])
    print(f"\n{'=' * 78}\nVERDICT ({best['variant']}): {v}\n{lab}\n{'=' * 78}")
    print(histogram(best["smoke"], best["fog"]))

    if len(results) > 1:
        print(f"\nSELECTION COST, DISCLOSED: {len(results)} variants evaluated on the same boxes. "
              f"{best['variant']} won, so its AUC is optimistically biased by roughly a "
              f"max-of-{len(results)} order statistic. It is VALIDATED at D11 on the untouched "
              f"CONFUSE arm — never by the number that selected it.")


if __name__ == "__main__":
    main()
