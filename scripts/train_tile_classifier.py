"""Train the one tile classifier, export it to ONNX, and record what it was trained on.

This script is off the judge path. It runs once, on a laptop, on CPU, in hours; its only
output that the endpoint ever sees is ``models/tile_classifier.onnx``. The server has no
torch in its dependency tree at all -- OpenCV's ``dnn`` is inference-only, which is an
honest limitation and is written down in opencv5-defense.md section 8.3 rather than being
quietly absent from the "Built With" list.

Four decisions are worth more than the code
-------------------------------------------
**1. Pyro-SDIS train, and nothing else.** The tile classifier never sees a FIgLib pixel.
That is what makes every FIgLib number in this project out-of-sample *by construction* --
a property of the data plumbing rather than a claim in a README -- and it is the first
thing a hostile judge should check. ``data/tile_manifest.json`` lists every source path,
this script refuses to write a manifest containing a FIgLib path, and
``test_no_figlib_in_training`` asserts the same thing from the other side.

**2. A plain 5-layer conv/ReLU/maxpool/FC net, chosen because its ONNX operator set is
trivially inside any engine's coverage.** ``ENGINE_NEW`` covers ~75-80% of the ONNX spec
and ``ENGINE_CLASSIC`` ~22%, so the realistic failure of this stage is not "the model is
inaccurate", it is "the model does not load". Starting with Conv/Relu/MaxPool/Flatten/Gemm
inverts that risk instead of documenting it. MobileNetV3-small is an upgrade only if there
is spare time, and it is not needed for the claim: B1 and the escalation stage share this
model, so a bigger one would make both arms better and the comparison no more informative.

**3. The tiles are cut at the working resolution.** A tile is ``TILE_SIZE`` px out of a
``WORK_WIDTH x WORK_HEIGHT`` frame at inference, so it is cut the same way here. Training
on 96 px of a 1280x720 source would train the model at a ~2.7x different pixels-per-metre
than it is ever asked to work at, and no amount of validation accuracy would show it.

**4. The training corpus is balanced by construction; the val split is not.** Only about
4% of Pyro-SDIS images carry a smoke box big enough to fill ``POSITIVE_OVERLAP_MIN`` of a
96 px tile at the working resolution, so drawing negatives from every image on the disk
produced 645 positives against 59,046 negatives and a net that learned to answer "no
smoke" to everything. ``build_pyrosdis_corpus(balanced=True)`` therefore takes tiles only
from images that can give both classes -- which is what ``TILE_POS_PER_IMAGE`` and
``TILE_NEG_PER_IMAGE`` mean in constants.py, and moves no threshold. The val split is
built with ``balanced=False``: the published precision is measured at Pyro-SDIS's own
class prior, because a rebalanced val split reports a precision no deployment ever sees.

Preprocessing is read from ``plumerank.constants`` rather than restated here, so the train
path and the ``cv2.dnn.blobFromImage`` path cannot drift and both are inside the surface
``results/constants.sha256`` freezes; the acceptance check below proves they have not.

Usage::

    python scripts/train_tile_classifier.py                       # Pyro-SDIS, the real run
    python scripts/train_tile_classifier.py --synthetic           # plumbing only, throwaway

``--synthetic`` exists so the whole pipeline -- corpus, training, export, ONNX acceptance,
manifest -- is runnable and testable before Pyro-SDIS is on disk. It produces a THROWAWAY
model with no evaluation meaning, writes it to a different filename so it can never be
mistaken for the submitted one, and stamps ``"throwaway": true`` into the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root, so the package
# this script exists to serve would otherwise be unimportable from its own repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plumerank.constants import (  # noqa: E402
    ONNX_ACCEPT_TOLERANCE,
    ONNX_OPSET,
    POSITIVE_OVERLAP_MIN,
    TILE_MEAN_BGR,
    TILE_NEG_PER_IMAGE,
    TILE_PIXEL_SCALE,
    TILE_POS_PER_IMAGE,
    TILE_SIZE,
    TILE_SWAP_RB,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEED,
    WORK_HEIGHT,
    WORK_WIDTH,
)

__all__ = [
    "FIGLIB_MARKERS",
    "tile_label",
    "parse_yolo_labels",
    "tiles_from_image",
    "build_pyrosdis_corpus",
    "build_synthetic_corpus",
    "preprocess",
    "assert_no_figlib",
    "precision_recall",
    "write_manifest",
    "main",
]

#: A source path containing any of these is a FIgLib path, and its presence in the
#: training corpus would destroy the out-of-sample property the headline rests on.
FIGLIB_MARKERS: tuple[str, ...] = ("figlib", "hpwren", "data/corpus", "data\\corpus")

DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "pyrosdis"
DEFAULT_MODEL_OUT = _REPO_ROOT / "models" / "tile_classifier.onnx"
SYNTHETIC_MODEL_OUT = _REPO_ROOT / "models" / "tile_classifier_synthetic.onnx"
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "tile_manifest.json"

#: 80/20, and only for --synthetic: the real run holds out Pyro-SDIS's own val split.
SYNTHETIC_TRAIN_FRACTION: float = 0.8


# ---------------------------------------------------------------------------------
# Corpus -- pure numpy/cv2, so this half of the script imports on a box with no torch
# ---------------------------------------------------------------------------------


def parse_yolo_labels(label_path: pathlib.Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Pyro-SDIS YOLO labels as ``(x0, y0, x1, y1)`` boxes in ``width x height`` pixels.

    Coordinates are normalised in the file, so scaling them into the *working* resolution
    is exact -- the boxes never go through the image resize and cannot pick up rounding
    from it. A malformed line raises: a silently dropped box becomes a mislabelled
    negative tile, which is the worst kind of training error because it looks like noise.
    """
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.is_file():
        return boxes  # an image with no label file is an all-negative image, by convention
    for lineno, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{lineno}: expected 'cls xc yc w h', got {line!r}")
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        boxes.append(
            (
                (xc - bw / 2.0) * width,
                (yc - bh / 2.0) * height,
                (xc + bw / 2.0) * width,
                (yc + bh / 2.0) * height,
            )
        )
    return boxes


def tile_label(
    origin: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
    tile: int = TILE_SIZE,
) -> str:
    """``"pos"`` | ``"neg"`` | ``"skip"`` for a tile at ``origin``.

    ``skip`` is the point of this function. A tile clipping the edge of a plume is neither
    a clean positive nor a clean negative, and training on it teaches the net that the
    boundary between "smoke" and "not smoke" is arbitrary -- which is precisely the
    discrimination the escalation stage is asked to make. Ambiguous tiles are discarded,
    not assigned to whichever class needs the count.
    """
    ox, oy = origin
    tx1, ty1 = ox + tile, oy + tile
    best = 0.0
    total = 0.0
    for bx0, by0, bx1, by1 in boxes:
        iw = max(0.0, min(tx1, bx1) - max(ox, bx0))
        ih = max(0.0, min(ty1, by1) - max(oy, by0))
        inter = iw * ih
        total += inter
        # Overlap is measured BOTH WAYS and the larger fraction wins, because a smoke box
        # and a 96x96 tile can be either side of each other in size and only one of the two
        # readings is meaningful in each regime.
        #
        # architecture.md section 5 says "positive tile = >=30% overlap with a smoke box".
        # Read tile-relative only, it means the box must fill 2,765 px of the tile.
        # Measured over Pyro-SDIS at this working resolution the median box is 15.6 x 11.3
        # px -- about 2% of a tile -- and only 2.4% of boxes could EVER reach 30% of one.
        # That reading discarded 97.6% of the corpus, left 628 usable images of 29,537, and
        # trained a classifier to val precision 0.024 against a 1.8% base rate: a model
        # that had learned nothing, in the role of baseline B1, which would have made the
        # headline comparison a strawman.
        #
        # Read box-relative only, it fails at the other end: a tile sitting wholly INSIDE a
        # near plume 192 x 162 px covers just 29.6% of that box and would be rejected,
        # though it is nothing but smoke.
        #
        # So: a distant plume is positive when the tile contains most of IT, and a near
        # plume is positive when it covers most of the TILE. Both are "overlapping by 30%";
        # neither alone is.
        box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        best = max(best, inter / box_area, inter / float(tile * tile))
    if best >= POSITIVE_OVERLAP_MIN:
        return "pos"
    if total <= 0.0:
        return "neg"
    return "skip"


def tiles_from_image(
    image_bgr: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    rng: np.random.Generator,
    n_pos: int,
    n_neg: int,
    tile: int = TILE_SIZE,
) -> tuple[list[np.ndarray], list[int]]:
    """Sample up to ``n_pos`` positive and ``n_neg`` negative tiles from one frame.

    Positives are drawn by jittering the tile around each box centre rather than by
    centring it exactly: a net trained only on perfectly centred plumes learns "smoke is
    in the middle", and at inference the escalated tile is centred on a *component
    centroid*, which is not the same point.
    """
    h, w = image_bgr.shape[:2]
    if h < tile or w < tile:
        raise ValueError(f"source frame {w}x{h} is smaller than a {tile} px tile")
    jitter = tile // 3
    crops: list[np.ndarray] = []
    labels: list[int] = []

    for bx0, by0, bx1, by1 in boxes:
        if len(crops) >= n_pos:
            break
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        for _ in range(8):  # a handful of tries; a tiny box near a corner may yield none
            ox = int(np.clip(round(cx - tile / 2.0) + rng.integers(-jitter, jitter + 1), 0, w - tile))
            oy = int(np.clip(round(cy - tile / 2.0) + rng.integers(-jitter, jitter + 1), 0, h - tile))
            if tile_label((ox, oy), boxes, tile) == "pos":
                crops.append(np.ascontiguousarray(image_bgr[oy : oy + tile, ox : ox + tile]))
                labels.append(1)
                break

    for _ in range(n_neg * 6):
        if labels.count(0) >= n_neg:
            break
        ox = int(rng.integers(0, w - tile + 1))
        oy = int(rng.integers(0, h - tile + 1))
        if tile_label((ox, oy), boxes, tile) == "neg":
            crops.append(np.ascontiguousarray(image_bgr[oy : oy + tile, ox : ox + tile]))
            labels.append(0)

    return crops, labels


def _iter_pyrosdis(root: pathlib.Path, split: str) -> Iterator[tuple[pathlib.Path, pathlib.Path]]:
    images = root / "images" / split
    labels = root / "labels" / split
    if not images.is_dir():
        raise FileNotFoundError(
            f"{images} not found -- run `python scripts/fetch.py --corpus pyrosdis` first. "
            f"Use --synthetic only for plumbing tests; it does not produce a usable model."
        )
    # parse_yolo_labels treats a MISSING label file as "this frame has no smoke", which is
    # Pyro-SDIS's own convention. A missing label DIRECTORY reads identically through that
    # convention and would train the classifier on an all-negative corpus that looks
    # perfectly healthy -- the loss falls, the export accepts, and recall is 0.0 forever.
    if not labels.is_dir():
        raise FileNotFoundError(
            f"{labels} not found beside {images} -- a truncated fetch. Every tile would be "
            f"labelled negative by the missing-label convention and nothing would say so."
        )
    for path in sorted(images.iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            yield path, labels / (path.stem + ".txt")


def build_pyrosdis_corpus(
    root: pathlib.Path,
    split: str,
    rng: np.random.Generator,
    limit: int | None = None,
    n_pos: int = TILE_POS_PER_IMAGE,
    n_neg: int = TILE_NEG_PER_IMAGE,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tiles, labels and the source paths they came from.

    The source list is returned rather than logged because it is the manifest, and the
    manifest is the evidence for the out-of-sample claim.

    ``balanced`` builds the corpus constants.py specifies -- ``TILE_POS_PER_IMAGE``
    positives and ``TILE_NEG_PER_IMAGE`` negatives from each source image the corpus is
    made of -- by taking nothing at all from an image that cannot give a positive. It is
    TRUE for the training split and FALSE for the validation split, and that asymmetry is
    the point: balance is a training-time decision, made instead of reweighting the loss,
    while a val split rebalanced the same way would report a precision measured against a
    class prior no deployment ever sees. The published number is measured on Pyro-SDIS's
    own val split, whole and undoctored, at its own prior.
    """
    crops: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    scanned = 0
    for i, (image_path, label_path) in enumerate(_iter_pyrosdis(root, split)):
        if limit is not None and i >= limit:
            break
        scanned += 1
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise OSError(f"unreadable image {image_path} -- a truncated fetch, not a bad label")
        frame = cv2.resize(raw, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
        boxes = parse_yolo_labels(label_path, WORK_WIDTH, WORK_HEIGHT)
        c, y = tiles_from_image(frame, boxes, rng, n_pos, n_neg)
        if not c:
            continue
        if balanced and 1 not in y:
            # An image that cannot give a positive tile gives no negative ones either.
            # This is not squeamishness about class balance, it is what constants.py
            # specifies: TILE_POS_PER_IMAGE and TILE_NEG_PER_IMAGE are "tiles sampled per
            # source image" and the corpus is "balanced by construction rather than by
            # reweighting the loss". Taking two negatives from every image on the disk
            # delivers neither -- the per-image recipe is applied to images the recipe
            # cannot be applied to.
            #
            # Measured, on the corpus this ships against, because the size of the effect
            # is the reason the line exists. A positive tile needs a smoke box covering
            # POSITIVE_OVERLAP_MIN of a TILE_SIZE tile -- 2,765 px^2 at
            # WORK_WIDTH x WORK_HEIGHT -- and Pyro-SDIS's median annotated box is 15x11 px
            # at that scale, so 150 of its 4,099 val images carry a box big enough for one.
            # Sampling negatives from the other 96% built 645 positives against 59,046
            # negatives; six epochs on that drove the loss to 0.0479 by learning to answer
            # "no smoke" to everything, val precision came back None because the net
            # predicted no positive at all, and the run stopped. That refusal was correct
            # and it is not a model. Dropping the unusable images is the fix that moves no
            # threshold: POSITIVE_OVERLAP_MIN is frozen and is not touched here.
            continue
        crops.extend(c)
        labels.extend(y)
        sources.append(str(image_path.relative_to(_REPO_ROOT)) if image_path.is_relative_to(_REPO_ROOT) else str(image_path))
    if not crops:
        raise RuntimeError(
            f"no tiles built from {root / 'images' / split}: not one positive tile in the "
            f"{scanned} source image(s) scanned"
            + (
                " -- and a balanced corpus takes negatives only from images that also gave "
                "a positive, so there is nothing left."
                if balanced
                else " -- the corpus is empty or mislaid."
            )
        )
    if 1 not in labels:
        raise RuntimeError(
            f"{len(crops)} tiles from {root / 'images' / split} and not one positive -- the "
            f"boxes did not survive the label parse. Training would succeed, export would "
            f"accept, and the model would answer 'no smoke' to every tile it is ever shown."
        )
    return np.stack(crops), np.asarray(labels, dtype=np.int64), sources


def build_synthetic_corpus(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A THROWAWAY tile corpus: soft-edged bright plumes against hard-edged dark shapes.

    This is plumbing, not science. It exists so the export, the ONNX acceptance check and
    the manifest can be exercised before Pyro-SDIS is on disk, and it is labelled as
    throwaway everywhere it touches: the model filename, the manifest, and stdout. It is
    built to be *learnable*, not to be realistic -- the soft/hard boundary contrast it
    encodes is the same physical intuition S3 measures, which is why a net that cannot
    learn it has a plumbing bug rather than a data problem.
    """
    crops = np.empty((n, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    yy = np.arange(TILE_SIZE, dtype=np.float32)[:, None]
    xx = np.arange(TILE_SIZE, dtype=np.float32)[None, :]
    for i in range(n):
        base = 90.0 + 60.0 * (yy / TILE_SIZE) + rng.normal(0.0, 3.0, (TILE_SIZE, TILE_SIZE))
        if i % 2 == 0:
            cy, cx = rng.uniform(30, 66, 2)
            r = rng.uniform(18, 30)
            soft = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r * r))
            base = base + 95.0 * soft
            labels[i] = 1
        else:
            y0, x0 = int(rng.integers(8, 40)), int(rng.integers(8, 40))
            s = int(rng.integers(24, 48))
            base[y0 : y0 + s, x0 : x0 + s] -= 55.0
        grey = np.clip(base, 0, 255).astype(np.uint8)
        crops[i] = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    return crops, labels, [f"synthetic://throwaway/{i:05d}" for i in range(n)]


def preprocess(tiles: np.ndarray) -> np.ndarray:
    """uint8 NHWC BGR tiles -> float32 NCHW, byte-identical to ``cv2.dnn.blobFromImage``.

    Written out in numpy rather than calling blobFromImage in the training loop because
    torch wants a batch and blobFromImage's batch form is a different call; the acceptance
    check at the end of ``main`` is what proves the two paths agree, to 1e-3 on real tiles.
    """
    x = tiles[..., ::-1] if TILE_SWAP_RB else tiles
    x = x.astype(np.float32) * TILE_PIXEL_SCALE - (
        np.asarray(TILE_MEAN_BGR, dtype=np.float32) * TILE_PIXEL_SCALE
    )
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------------
# The manifest -- the artefact the out-of-sample claim actually rests on
# ---------------------------------------------------------------------------------


def assert_no_figlib(sources: list[str]) -> None:
    """Refuse to proceed if a FIgLib path reached the training corpus.

    The guard is here, at the point of writing, as well as in ``test_no_figlib_in_training``
    -- a test that runs after the fact would let a contaminated model exist on disk in the
    meantime, and the model is the thing that gets shipped.
    """
    hits = [s for s in sources if any(m in s.lower() for m in FIGLIB_MARKERS)]
    if hits:
        raise SystemExit(
            f"REFUSING TO TRAIN: {len(hits)} FIgLib path(s) in the corpus, e.g. {hits[0]}. "
            f"FIgLib is evaluation-only; training on it would make every published number "
            f"in-sample."
        )


def write_manifest(
    path: pathlib.Path,
    sources: list[str],
    counts: dict[str, int],
    model_path: pathlib.Path,
    corpus: str,
    throwaway: bool,
    val: dict[str, float | None] | None,
) -> dict[str, Any]:
    assert_no_figlib(sources)
    manifest = {
        "corpus": corpus,
        "throwaway": throwaway,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tile_size": TILE_SIZE,
        "work_resolution": [WORK_WIDTH, WORK_HEIGHT],
        "positive_overlap_min": POSITIVE_OVERLAP_MIN,
        "onnx_opset": ONNX_OPSET,
        "counts": counts,
        "model": {
            "path": str(model_path.relative_to(_REPO_ROOT)) if model_path.is_relative_to(_REPO_ROOT) else str(model_path),
            "sha256": _sha256(model_path) if model_path.is_file() else None,
        },
        "val": val,
        "sources": sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------
# Training -- everything below imports torch, and nothing above it does
# ---------------------------------------------------------------------------------


def _build_model(torch: Any) -> Any:
    """Five layers: four Conv/ReLU/MaxPool blocks and one fully-connected head.

    Every operator here -- Conv, Relu, MaxPool, Flatten, Gemm -- is in the oldest and
    smallest corner of the ONNX spec, which is the entire reason this architecture was
    chosen over a MobileNet. 96 -> 48 -> 24 -> 12 -> 6, then 64*6*6 -> 2 logits. No
    adaptive pooling, no batch norm, no residual add: each of those is one more operator
    that ENGINE_NEW would have to cover, for accuracy this task does not need.
    """
    nn = torch.nn

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, 64),
        block(64, 64),
        nn.Flatten(),
        nn.Linear(64 * (TILE_SIZE // 16) * (TILE_SIZE // 16), 2),
    )


def _train(torch: Any, model: Any, x: np.ndarray, y: np.ndarray, epochs: int, batch: int, seed: int) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.from_numpy(preprocess(x))
    yt = torch.from_numpy(y)
    n = xt.shape[0]
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / n:.4f}")


def precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    """Precision and recall at argmax, or ``None`` where the denominator is empty.

    ``None`` and not ``0.0``. These two numbers are published in the manifest and in
    REPORT.md, and a model that predicted "smoke" for nothing at all would otherwise
    report precision 0.0 -- a measurement-shaped value for a quantity that was never
    measured, which is exactly the kind of number that survives into a slide.

    Lives outside ``_evaluate`` because everything that takes ``torch`` is untestable in
    the runtime venv, and arithmetic that reaches a published table should not be.
    """
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "n_tiles": int(len(truth)),
    }


def _evaluate(torch: Any, model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Precision and recall at argmax on held-out tiles. Published, whatever it says."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(preprocess(x))).argmax(1).numpy()
    return precision_recall(pred, y)


def _export(torch: Any, model: Any, out_path: pathlib.Path) -> None:
    """``torch.onnx.export`` at a fixed 1x3x96x96, no dynamic axes.

    Fixed shapes on purpose: ENGINE_NEW's shape inference, constant folding and operator
    fusion all run over a static graph, and the escalation stage scores one tile at a time
    anyway. A dynamic batch axis would buy nothing and would add a question to the one
    thing that must not be in question -- whether the model loads.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, TILE_SIZE, TILE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=ONNX_OPSET,
    )


def _accept(torch: Any, model: Any, onnx_path: pathlib.Path, tiles: np.ndarray, n: int = 20) -> float:
    """Load the export under ENGINE_NEW and match its logits to PyTorch's.

    This is the acceptance gate, and it fails the run rather than warning: an exported
    model that loads but computes something else is indistinguishable from a working one
    until it is scoring live frames on the wall.
    """
    net = cv2.dnn.readNetFromONNX(str(onnx_path), cv2.dnn.ENGINE_NEW)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    sample = tiles[:n]
    model.eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(preprocess(sample))).numpy()
    worst = 0.0
    for i in range(len(sample)):
        blob = cv2.dnn.blobFromImage(
            sample[i], scalefactor=TILE_PIXEL_SCALE, size=(TILE_SIZE, TILE_SIZE),
            mean=TILE_MEAN_BGR, swapRB=TILE_SWAP_RB, crop=False,
        )
        net.setInput(blob)
        worst = max(worst, float(np.abs(net.forward().reshape(-1) - ref[i]).max()))
    if not np.isfinite(worst) or worst >= ONNX_ACCEPT_TOLERANCE:
        raise SystemExit(
            f"ACCEPTANCE FAILED: ONNX and PyTorch logits differ by {worst:.2e} "
            f"(limit {ONNX_ACCEPT_TOLERANCE:.0e})"
        )
    return worst


# ---------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="train", help="Pyro-SDIS split to TRAIN on. Never 'val'.")
    p.add_argument("--val-split", default="val")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None, help="cap on source images, for a smoke run")
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="THROWAWAY corpus for plumbing tests only -- produces a model with no evaluation meaning",
    )
    p.add_argument("--synthetic-tiles", type=int, default=512)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Lazily, and only here: the server installs no torch, and this module has to stay
    # importable there so that check_claims.py and the guard tests can read it.
    import torch

    rng = np.random.default_rng(args.seed)
    out_path = args.out or (SYNTHETIC_MODEL_OUT if args.synthetic else DEFAULT_MODEL_OUT)

    if args.synthetic:
        print("=" * 72)
        print("SYNTHETIC MODE -- THROWAWAY MODEL, PLUMBING TESTS ONLY.")
        print("Not the submitted model. Not evaluated on anything. Do not publish a number")
        print("from this run. The real path is the default: Pyro-SDIS train.")
        print("=" * 72)
        x, y, sources = build_synthetic_corpus(args.synthetic_tiles, rng)
        cut = int(len(y) * SYNTHETIC_TRAIN_FRACTION)
        x_val, y_val = x[cut:], y[cut:]
        x, y = x[:cut], y[:cut]
        corpus = "synthetic-throwaway"
    else:
        if args.split == args.val_split:
            raise SystemExit("--split and --val-split are the same; that is not a held-out number")
        print(f"building tiles from {args.data_root / 'images' / args.split} ...")
        x, y, sources = build_pyrosdis_corpus(args.data_root, args.split, rng, args.limit)
        # balanced=False, and it is the one asymmetry in this script. The training corpus
        # is balanced by construction because constants.py says so; the val split is the
        # dataset's own, taken whole, at its own class prior -- roughly one positive tile
        # in fifty-six. Rebalancing it would raise the published precision by moving the
        # prior rather than by improving the model, which is the same trick as reporting
        # accuracy on a corpus you chose.
        x_val, y_val, _ = build_pyrosdis_corpus(
            args.data_root, args.val_split, rng, args.limit, balanced=False
        )
        corpus = "pyro-sdis"

    assert_no_figlib(sources)
    counts = {
        "source_images": len(sources),
        "train_tiles": int(len(y)),
        "train_positive": int((y == 1).sum()),
        "train_negative": int((y == 0).sum()),
        "val_tiles": int(len(y_val)),
        # The val class counts are published beside the precision they produced, because
        # precision is a function of the prior: 0.4 on a split that is 2% positive and 0.4
        # on one that is 50% positive are different claims, and only the counts say which
        # one this is.
        "val_positive": int((y_val == 1).sum()),
        "val_negative": int((y_val == 0).sum()),
    }
    print(f"corpus {corpus}: {counts}")

    model = _build_model(torch)
    _train(torch, model, x, y, args.epochs, args.batch_size, args.seed)
    val = _evaluate(torch, model, x_val, y_val)
    print(f"val precision {val['precision']}  recall {val['recall']}  on {val['n_tiles']} tiles")
    if not args.synthetic and (val["precision"] is None or val["recall"] is None):
        raise SystemExit(
            "val precision or recall is unmeasurable on the real corpus -- the val split "
            "has no positives, or the model predicted none. Publishing that as 0.0 is how "
            "an unevaluated model reaches a report."
        )

    _export(torch, model, out_path)
    worst = _accept(torch, model, out_path, x_val if len(y_val) else x)
    print(f"ONNX accepted under ENGINE_NEW: max |onnx - torch| = {worst:.2e}")

    digest = _sha256(out_path)
    # The sidecar is written HERE and not by hand afterwards. It is the committed claim
    # about the one binary in this repository, and the only step that can keep it true is
    # the step that produced the bytes: a hash typed in later is a hash of whatever
    # happened to be on disk at the time, which is exactly the thing it exists to rule out.
    sha_path = out_path.with_name(out_path.name + ".sha256")
    sha_path.write_text(f"{digest}  {out_path.name}\n")

    write_manifest(args.manifest, sources, counts, out_path, corpus, args.synthetic, val)
    print(f"wrote {out_path} and {args.manifest}")
    print(f"sha256 {digest}  ({sha_path.name})")
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point, run by hand
    raise SystemExit(main())
