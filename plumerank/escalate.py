"""The ~1% learned path -- ONE model, TWO uses, which is what makes the headline a
controlled experiment rather than a contest between two different systems.

The single most attackable thing a project like this can do is beat a baseline by
bringing a bigger model to the fight. So there is no second model here. The same
``models/tile_classifier.onnx``, the same weights, the same preprocessing and the same
``cv2.dnn`` call path serve both arms:

* **Use 1 -- baseline B1.** ``score_frame_grid`` runs the classifier over a tile grid
  covering the whole frame and takes the max. That is the honest per-frame learned
  answer, on every frame, with full coverage. It is not hobbled.
* **Use 2 -- escalation.** ``score_tile`` runs the same classifier over exactly ONE tile,
  around the component the classical statistic already found, on the ~1% of frames
  ``should_escalate`` selects. Its score folds into the hazard.

Because the learned component is *identical* in both arms, PlumeRank cannot win by
having a bigger model. Every measured difference between the arms is attributable to the
classical anchoring statistic and to the budget calibration -- which is the claim the
project exists to make. Two systems that differ in several ways at once cannot support
that claim; two systems that differ in exactly one way can.

Why the invocation counter is a first-class part of the interface
-----------------------------------------------------------------
``invocations`` counts **every** forward pass, in both uses, and ``bench.py`` publishes it
per camera-night for both arms. The 99/1 cost architecture is the project's economic
argument, and an economic argument that is asserted in a README rots the first time
someone adds a second forward pass. Counting it means the claim is measured. B1 costs
``frames x grid_tiles`` invocations; PlumeRank costs ``escalated_frames x 1``. Neither
number is written down here -- both are printed by the benchmark.

Scale, and why no crop is ever resized
--------------------------------------
Training tiles are cut at ``TILE_SIZE`` out of a ``WORK_WIDTH x WORK_HEIGHT`` frame, so
inference cuts them the same way and ``blobFromImage``'s resize is a no-op. Feeding the
net a 200 px crop squeezed into 96 px would present it a different pixels-per-metre than
every tile it was trained on. The cost of that choice is a named limitation, not a hidden
one: a component larger than one tile is scored on the tile at its centroid.

OpenCV 5 surfaces used here (the OpenCV 5 design notes section 5): ``cv2.dnn.readNetFromONNX``
with the ``engine`` argument, ``cv2.dnn.ENGINE_NEW``, ``cv2.dnn.blobFromImage``,
``cv2.dnn.Net.forward``. ``ENGINE_NEW`` is 5.x-only and CPU-only in 5.0, which is free on
Graviton and is written down as a cap on the architecture in the OpenCV 5 design notes section 8.2.
``Net::setPreferableBackend`` is called here too (build-contract.md W4 names it) and is
**used and not counted** as depth, in the same sense as ``goodFeaturesToTrack`` in the
stabiliser: it selects a backend, it does not do analysis. It is deliberately NOT one of
the four counted ``dnn`` surfaces in the OpenCV 5 design notes section 5, and it belongs on that
file's section 6a anti-inflation table alongside the four drawing calls.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

from .constants import (
    B1_GRID_STRIDE,
    ESCALATION_HAZARD_MIN,
    ESCALATION_TOP_K,
    TILE_MEAN_BGR,
    TILE_PIXEL_SCALE,
    TILE_SIZE,
    TILE_SWAP_RB,
)

__all__ = ["TileClassifier", "should_escalate", "grid_origins"]


def grid_origins(extent: int, tile: int = TILE_SIZE, stride: int = B1_GRID_STRIDE) -> list[int]:
    """Tile origins along one axis, with the last tile flush against the far edge.

    Args:
        extent: length of the axis in pixels.
        tile: tile side in pixels.
        stride: step between origins.

    Returns:
        Ascending origins, each in ``[0, extent - tile]``.

    Raises:
        ValueError: if the axis is shorter than one tile, or ``stride`` is not positive,
            or ``stride`` exceeds ``tile`` (which would put gaps *between* tiles rather
            than only at the far edge).

    ``range(0, extent - tile + 1, stride)`` alone leaves an unscanned band of up to
    ``stride - 1`` px at the far edge, and a plume that first appears in that band would
    be invisible to B1 -- which would make B1 a strawman and the headline worthless. The
    flush final origin overlaps its neighbour instead. This function is public because
    B1's published cost is ``len(grid_origins(W)) * len(grid_origins(H))`` invocations per
    frame, so the geometry is part of the claim and is tested directly.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > tile:
        raise ValueError(
            f"stride {stride} exceeds tile {tile}: the grid would leave gaps between "
            f"tiles, and a plume in one of them would be invisible to B1"
        )
    if extent < tile:
        raise ValueError(f"axis of {extent} px cannot hold a {tile} px tile")

    last = extent - tile
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def should_escalate(rank: int, hazard: float) -> bool:
    """Whether this camera-frame earns a forward pass. Two constants, one function.

    Args:
        rank: 1-based rank in this tick's watch list.
        hazard: the classical fused hazard, in [0, 1].

    Both conditions are load-bearing. ``rank`` bounds the cost STRUCTURALLY -- at most
    ``ESCALATION_TOP_K`` of the roster can escalate in any tick, whatever the hazards do,
    and no calibration can loosen that. ``hazard`` is what makes the realised fraction ~1%
    rather than the structural ceiling, because a quiet camera's hazard is
    ``sigmoid(FUSION_BIAS)`` and never approaches the floor.

    **Corrected 2026-09-06.** This docstring used to end: "An adaptive rule would make the
    escalation fraction a fitted quantity, and the whole point is that it is measured." That
    is no longer true of the floor and the sentence is removed rather than left standing.
    ``ESCALATION_HAZARD_MIN`` is now SOLVED on the CONFUSE arm against a 1% design target
    (``scripts/fit_escalation_floor.py``), because an absolute floor does not survive a refit
    of the fusion: the a-priori 0.35 put the floor *below* a quiet camera once the weights
    were fitted, and the gate silently stopped gating.

    What the cost argument still rests on, unchanged: the rank half of the gate is
    structural; the floor is calibrated on the NON-FIRE arm and never against fire windows;
    and the fraction realised on any fire arm is measured and published, never targeted.
    """
    return rank <= ESCALATION_TOP_K and hazard >= ESCALATION_HAZARD_MIN


class TileClassifier:
    """The one learned component, executed by ``cv2.dnn`` on OpenCV 5's new engine.

    Args:
        onnx_path: path to the exported classifier.
        engine: ``cv2.dnn.ENGINE_NEW`` by default. ``ENGINE_CLASSIC`` covers ~22% of the
            ONNX spec against ``ENGINE_NEW``'s ~75-80%, so the argument exists to be
            *reported* by ``/build-info``, not to be quietly downgraded on failure.

    Raises:
        FileNotFoundError: the model is missing. Escalation is a judged capability, so a
            missing model is a hard stop and never a silent 0.0 score.
    """

    __slots__ = ("_path", "_net", "_engine", "_invocations")

    def __init__(self, onnx_path: str | pathlib.Path, engine: int = cv2.dnn.ENGINE_NEW) -> None:
        self._path = pathlib.Path(onnx_path)
        if not self._path.is_file():
            raise FileNotFoundError(
                f"tile classifier not found at {self._path} -- run "
                f"scripts/train_tile_classifier.py; escalation does not degrade to a constant"
            )
        self._engine = int(engine)
        self._net = cv2.dnn.readNetFromONNX(str(self._path), self._engine)
        # No target is set: ENGINE_NEW is CPU-only in 5.0 (the OpenCV 5 design notes 8.2), so
        # asking for a non-CPU target here would be a request the engine cannot honour
        # and would only make /build-info's reported backend disagree with reality.
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._invocations = 0

    @property
    def invocations(self) -> int:
        """Forward passes since construction, across BOTH uses.

        One counter and not two, deliberately: a per-use counter would let the 99/1
        arithmetic be quoted against whichever denominator flattered it.
        """
        return self._invocations

    @property
    def engine(self) -> int:
        return self._engine

    @property
    def path(self) -> pathlib.Path:
        return self._path

    # -----------------------------------------------------------------------------
    # Use 2 -- escalation, ~1% of frames
    # -----------------------------------------------------------------------------

    def score_tile(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """P(smoke) for ONE tile centred on ``bbox``. Exactly one forward pass.

        Args:
            frame: BGR working-resolution frame.
            bbox: ``(x, y, w, h)`` of the classical component, in frame pixels.

        Returns:
            Probability in [0, 1].

        Raises:
            ValueError: the frame is not BGR uint8, is smaller than one tile, or the
                bbox is degenerate; or the net returned a non-finite output.

        The tile is ``TILE_SIZE`` square regardless of the component's size and is slid
        inside the frame rather than padded at the border: a padded tile would present the
        net a constant band no training tile contains, and a border component is exactly
        where a false negative is most expensive.
        """
        _check_frame(frame)
        x, y, w, h = (int(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"degenerate bbox {bbox!r} -- nothing to score")
        cx = x + w / 2.0
        cy = y + h / 2.0
        ox = _clamp_origin(int(round(cx - TILE_SIZE / 2.0)), frame.shape[1])
        oy = _clamp_origin(int(round(cy - TILE_SIZE / 2.0)), frame.shape[0])
        return self._forward(frame[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE])

    # -----------------------------------------------------------------------------
    # Use 1 -- baseline B1, every frame
    # -----------------------------------------------------------------------------

    def score_frame_grid(self, frame: np.ndarray) -> float:
        """B1: the max tile score over a grid covering the whole frame.

        Args:
            frame: BGR working-resolution frame.

        Returns:
            The maximum probability over the grid, in [0, 1].

        Raises:
            ValueError: the frame is not BGR uint8, is smaller than one tile, or the net
                returned a non-finite output on any tile.

        Max and not mean: B1 answers "is there smoke anywhere in this frame", and a mean
        over ~45 tiles would let 44 empty tiles bury the one that matters -- a strawman
        baseline dressed as a fair one. Each tile is its own ``forward`` rather than one
        batched blob so that ``invocations`` counts the work B1 actually does; batching
        would report 1 where the arithmetic in the architecture notes section 6 counts *g*.
        """
        _check_frame(frame)
        xs = grid_origins(frame.shape[1])
        ys = grid_origins(frame.shape[0])
        best = 0.0
        for oy in ys:
            for ox in xs:
                best = max(best, self._forward(frame[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE]))
        return best

    # -----------------------------------------------------------------------------

    def _forward(self, tile: np.ndarray) -> float:
        if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
            raise ValueError(
                f"tile is {tile.shape[1]}x{tile.shape[0]}, expected {TILE_SIZE} square -- "
                f"the frame is smaller than one tile and no resize is applied here"
            )
        blob = cv2.dnn.blobFromImage(
            tile,
            scalefactor=TILE_PIXEL_SCALE,
            size=(TILE_SIZE, TILE_SIZE),
            mean=TILE_MEAN_BGR,
            swapRB=TILE_SWAP_RB,
            crop=False,
        )
        self._net.setInput(blob)
        out = self._net.forward()
        self._invocations += 1
        return _probability(out)


def _check_frame(frame: np.ndarray) -> None:
    """Reject anything that is not a BGR uint8 frame, before it becomes a score.

    ``blobFromImage`` will happily scale a float32 frame that is already in [0, 1] by
    another 1/255 and hand the net a black tile, and the net will return a perfectly
    well-formed probability for it. Nothing downstream can tell that number from a real
    one, and it folds straight into the published hazard -- so the frame contract is
    checked here rather than documented and hoped for.
    """
    if frame.dtype != np.uint8:
        raise ValueError(
            f"frame dtype is {frame.dtype}, expected uint8 -- the preprocessing scale "
            f"({TILE_PIXEL_SCALE}) assumes a 0-255 range and would silently mis-scale it"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame shape is {frame.shape}, expected (h, w, 3) BGR -- the classifier was "
            f"trained on three channels and swapRB is meaningless without them"
        )


def _clamp_origin(origin: int, extent: int) -> int:
    if extent < TILE_SIZE:
        raise ValueError(f"frame axis of {extent} px cannot hold a {TILE_SIZE} px tile")
    return max(0, min(origin, extent - TILE_SIZE))


def _probability(out: np.ndarray) -> float:
    """Map the net's output to P(smoke), or refuse.

    Two heads are accepted -- one logit (sigmoid) and two logits (softmax, index 1) -- and
    anything else raises, as does a non-finite output from any head. Guessing at an unknown head would produce a plausible number
    from a model nobody has verified, which is the one failure mode this project cannot
    afford: the escalation score folds straight into the published hazard.
    """
    flat = np.asarray(out, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        # NaN is the expensive one. score_frame_grid takes max() over the grid, and
        # max(0.0, nan) is 0.0 in Python -- so a net that had gone NaN would report "no
        # smoke anywhere in this frame" with no warning, on every frame, forever.
        raise ValueError(
            f"tile classifier returned a non-finite output {flat!r} -- the model is "
            f"broken, and a probability derived from it would be a plausible wrong answer"
        )
    if flat.size == 1:
        return float(1.0 / (1.0 + np.exp(-flat[0])))
    if flat.size == 2:
        shifted = flat - flat.max()  # softmax overflows on large logits without this
        e = np.exp(shifted)
        return float(e[1] / e.sum())
    raise ValueError(
        f"tile classifier returned {flat.size} values; expected 1 logit or 2 class logits"
    )
