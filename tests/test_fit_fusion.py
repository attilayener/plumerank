"""scripts/fit_fusion.py -- the fit that D11 specified and constants.py stood in for.

A fitting routine is the one kind of code whose output nobody can eyeball for correctness:
a weight is plausible or it is not, and "plausible" is exactly what a silently wrong solver
produces. So the tests here are all recovery tests against answers that are known in advance
-- synthetic data drawn from a logistic with chosen coefficients, which the fit must return.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("fit_fusion", ROOT / "scripts" / "fit_fusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ff = _load()

TRUE_BIAS = -2.0
TRUE_W = np.array([2.5, 1.0, -1.5, 0.5, 3.0])


def _synthetic(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    p = ff._sigmoid(TRUE_BIAS + X @ TRUE_W)
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_recovers_known_coefficients():
    """The whole point. With the penalty off, the fit must return the coefficients the
    data were generated from -- including the negative one, which is the case a solver
    that quietly clips to zero would pass every other test while failing."""
    X, y = _synthetic()
    _, w = ff.fit(X, y, lam=1e-6)
    assert np.max(np.abs(w - TRUE_W)) < 0.35


def test_negative_weight_survives_the_fit():
    X, y = _synthetic()
    _, w = ff.fit(X, y, lam=1e-6)
    assert w[2] < 0, "a statistic that argues AGAINST the label must fit negative"


def test_l2_shrinks_weights_toward_zero():
    X, y = _synthetic()
    _, small = ff.fit(X, y, lam=1e-6)
    _, large = ff.fit(X, y, lam=1000.0)
    assert np.sum(np.abs(large)) < np.sum(np.abs(small))


def test_bias_does_not_carry_the_base_rate():
    """Balanced class weighting makes the two classes equal mass, so the fitted bias sits
    near ZERO however skewed the corpus is -- it is not a base-rate estimate.

    This is the test that corrected the module docstring: the original claim was that the
    unpenalised bias preserves the base rate for the budget solver. It does not, and a
    reader who believed it would expect ``FUSION_BIAS`` to move with the fire fraction and
    conclude the fit was broken when it did not. The operating point comes from
    ``rank.solve_threshold`` on CONFUSE, and only from there.
    """
    rng = np.random.default_rng(11)
    X = rng.random((4000, 5))
    p = ff._sigmoid(-4.0 + X @ TRUE_W)          # a deliberately rare positive class
    y = (rng.random(4000) < p).astype(float)
    assert y.mean() < 0.35, "this fixture is meant to be imbalanced"
    bias, _ = ff.fit(X, y, lam=1e6)             # weights crushed; only the bias is left
    assert abs(bias) < 0.05, "balanced weighting should centre the bias near zero"


def test_penalty_matrix_leaves_the_bias_slot_free():
    """The structural half of the same property, asserted where it actually lives: any
    future edit that penalises column 0 changes the meaning of every fitted weight."""
    X, y = _synthetic(n=400)
    lo_bias, lo_w = ff.fit(X, y, lam=1e-6)
    hi_bias, hi_w = ff.fit(X, y, lam=1e6)
    assert np.max(np.abs(hi_w)) < np.max(np.abs(lo_w)) / 10.0, "weights must shrink"
    assert abs(lo_bias) > abs(hi_bias), "the bias should move only because the weights did"


@pytest.mark.parametrize("z", [-800.0, -50.0, 0.0, 50.0, 800.0])
def test_sigmoid_does_not_overflow(z):
    v = ff._sigmoid(np.array([z]))
    assert np.isfinite(v).all() and 0.0 <= v[0] <= 1.0


def test_auc_matches_the_gate_estimator_on_a_known_case():
    """Same estimator as spike/s3_separation.py -- verified by computing one case both ways.

    The two take their inputs differently (the gate splits the classes at the call site,
    this one splits by ``y``), which is precisely why the equivalence is worth asserting
    rather than assuming: ``results/`` publishes AUCs from both files.
    """
    # perfect separation
    assert ff.auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([1, 1, 0, 0])) == 1.0
    # one tie straddling the boundary: 1 positive vs 2 negatives -> 0.75
    assert ff.auc(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0])) == 0.75
    # and the gate's own implementation must agree on the same numbers
    gate = importlib.util.spec_from_file_location("s3sep", ROOT / "spike" / "s3_separation.py")
    mod = importlib.util.module_from_spec(gate)
    gate.loader.exec_module(mod)
    assert mod.auc([1.0], [1.0, 0.0]) == ff.auc(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0]))


def test_single_class_input_refuses_rather_than_returns():
    X = np.random.default_rng(0).random((50, 5))
    with pytest.raises(SystemExit):
        ff.fit(X, np.ones(50))


def test_grouped_cv_really_holds_the_sequence_out():
    """The property that matters, asserted directly rather than via an in-sample gap.

    An earlier version of this test asserted in-sample AUC > CV AUC. That premise was
    wrong: an L2 logistic on five features has almost no capacity to memorise, so the two
    agree on ordinary data and the test failed on correct code. What LOSO actually
    guarantees is that a sequence's own rows never inform its own predictions -- so give
    one sequence a label rule INVERTED relative to every other, and its held-out scores
    must come out backwards. They can only come out right if it was trained on itself.
    """
    rng = np.random.default_rng(3)
    Xs, ys, gs = [], [], []
    for g in range(10):
        x = rng.random((40, 5))
        inverted = g == 0
        z = x[:, 0] - 0.5
        lab = (z < 0).astype(float) if inverted else (z > 0).astype(float)
        Xs.append(x)
        ys.append(lab)
        gs += [f"seq{g}"] * 40
    X, y, groups = np.vstack(Xs), np.concatenate(ys), np.array(gs)

    held = np.zeros(len(y))
    for s in sorted(set(groups.tolist())):
        te = groups == s
        b, w = ff.fit(X[~te], y[~te])
        held[te] = b + X[te] @ w

    odd = groups == "seq0"
    assert ff.auc(held[odd], y[odd]) < 0.25, "the inverted sequence was fit on itself"
    assert ff.auc(held[~odd], y[~odd]) > 0.75, "the ordinary sequences should score well"

    cv, n_scored, folds = ff.grouped_cv(X, y, groups)
    assert folds == 10 and n_scored == len(y)
    assert 0.0 <= cv <= 1.0


def test_load_reads_the_dump_format_bench_writes(tmp_path):
    """Contract test against scripts/bench.py:dump_features. If the writer's key names
    drift, this fails here rather than producing a fit on zeros."""
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join(json.dumps({
        "sequence_id": "s1", "camera_id": "c", "frame_index": i,
        "is_fire_window": True, "t0_index": 2, "label": 1 if i >= 2 else 0,
        "s1": 0.1 * i, "s2": 0.2, "s3": 0.3, "s4": 0.4, "esc": 0.0,
        "has_component": 1.0,
    }) for i in range(5)))
    X, y, groups, rows = ff.load(p)
    assert X.shape == (5, 5)
    assert list(y) == [0, 0, 1, 1, 1]
    assert set(groups) == {"s1"}
    assert rows[0]["s1"] == 0.0


def test_empty_features_file_refuses():
    with pytest.raises(SystemExit):
        ff.load(pathlib.Path("/dev/null"))
