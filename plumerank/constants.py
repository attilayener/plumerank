"""Every fitted number in PlumeRank, in one file, so that freezing it is a single act.

Why this file exists at all
---------------------------
The headline number is measured on 60 held-out sequences that are read exactly *once*,
after the constants are frozen. "Frozen" has to mean something a judge can check, so it
means this: the constants live in one module, ``freeze_hash()`` hashes its source, and
``results/constants.sha256`` records that hash. ``results/hold_access.log`` records every
read of the HOLD arm with a timestamp and a commit sha, and a test fails if HOLD was read
before the hash file existed.

Provenance of the numbers below
-------------------------------
* ``a priori``  -- chosen before any data was seen, from the physics or from a doc page.
                   Never fitted. Changing one is a spec change, not a tuning pass.
* ``fit on DEV`` -- fitted on the 30-sequence DEV arm, which exists to be fitted on.
* ``solved``     -- not a constant at all: computed by ``rank.solve_threshold`` on the
                   CONFUSE arm and written to ``results/``. Listed here only so that the
                   frozen surface is complete.

Nothing in this file is fitted on HOLD. That is the whole point of the file.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Final

# ---------------------------------------------------------------------------------
# Replay geometry -- a priori, from the corpus, not tunable
# ---------------------------------------------------------------------------------

#: HPWREN FIgLib publishes one frame per minute around each ignition.
SECONDS_PER_FRAME: Final[float] = 60.0
MINUTES_PER_FRAME: Final[float] = SECONDS_PER_FRAME / 60.0

#: Every window -- DEV, HOLD and CONFUSE alike -- is 81 frames: 40 before the ignition
#: clock, the ignition frame itself, and 40 after. A CONFUSE window is the same span on
#: the same camera at the same time of day, which is what makes "one camera-night"
#: comparable between the arms (seed-data.md section 4).
FRAMES_PER_WINDOW: Final[int] = 81
IGNITION_FRAME_INDEX: Final[int] = 40

#: One replay window is one camera-night for rate arithmetic. The alert budget is
#: expressed per camera-night because that is the unit a dispatcher actually feels.
MINUTES_PER_CAMERA_NIGHT: Final[float] = FRAMES_PER_WINDOW * MINUTES_PER_FRAME

#: Working resolution. FIgLib frames are 3072x2048; the pipeline runs at 480x270 so that
#: 40 concurrent MOG2 models fit in the 2 GiB of a t4g.small (spec.md section 11, ~11 MB
#: per camera -> ~432 MB) and so the classical path stays inside its 30 ms budget.
WORK_WIDTH: Final[int] = 480
WORK_HEIGHT: Final[int] = 270

# ---------------------------------------------------------------------------------
# stabilise.py -- a priori
# ---------------------------------------------------------------------------------

#: cv2.phaseCorrelate returns (shift, response). Below this response the phase peak is
#: not trustworthy -- typically a featureless sky with nothing to correlate -- and the
#: feature-based fallback runs instead. ONE branch, ONE threshold, no adaptive scheme.
PHASE_RESPONSE_MIN: Final[float] = 0.15

#: Fallback path: goodFeaturesToTrack -> calcOpticalFlowPyrLK -> estimateAffinePartial2D.
FALLBACK_MAX_CORNERS: Final[int] = 200
FALLBACK_QUALITY_LEVEL: Final[float] = 0.01
FALLBACK_MIN_DISTANCE: Final[int] = 8
FALLBACK_MIN_INLIERS: Final[int] = 12
USAC_REPROJ_THRESHOLD: Final[float] = 3.0
USAC_CONFIDENCE: Final[float] = 0.99
USAC_MAX_ITERS: Final[int] = 2000

#: A shift larger than this is not camera shake, it is a pan or a corrupt frame. The
#: frame is passed through unstabilised and flagged, rather than warped by a wrong number.
MAX_PLAUSIBLE_SHIFT_PX: Final[float] = 40.0

# ---------------------------------------------------------------------------------
# segment.py -- MOG2 defaults are OpenCV's own; the rest is a priori
# ---------------------------------------------------------------------------------

MOG2_HISTORY: Final[int] = 120
MOG2_VAR_THRESHOLD: Final[float] = 24.0
MOG2_DETECT_SHADOWS: Final[bool] = False  # shadows are a CONFUSE class, not a nuisance

#: Frames the background model needs before its output is meaningful. The UI has a
#: designed state for exactly this span (ui.md section 7, "warming background model").
WARMUP_FRAMES: Final[int] = 8

MORPH_OPEN_KERNEL: Final[int] = 3   # kill single-pixel speckle
MORPH_CLOSE_KERNEL: Final[int] = 7  # a plume is porous; close it into one component

CLAHE_CLIP_LIMIT: Final[float] = 2.0
CLAHE_TILE_GRID: Final[int] = 8

#: At 480x270 a component under this many pixels cannot carry a boundary band wide
#: enough for S3 to mean anything, so it is dropped before it reaches the statistic.
MIN_COMPONENT_AREA_PX: Final[int] = 60
MAX_COMPONENTS_PER_FRAME: Final[int] = 8

# ---------------------------------------------------------------------------------
# statistic.py -- THE PRIMITIVE. Six constants, and their provenance is the point.
# ---------------------------------------------------------------------------------

#: Rolling window, in frames, over which S1 and S2 are measured. 10 minutes: long enough
#: for a plume to visibly climb, short enough that the dispatcher's answer is not stale.
STAT_WINDOW_FRAMES: Final[int] = 10

#: S1 -- base anchoring. The lowest extent of a fire-tied component stays within +/-k px
#: while its top climbs. k is in WORK_HEIGHT pixels. a priori: it is the physical claim,
#: not a knob.
S1_ANCHOR_TOLERANCE_PX: Final[float] = 4.0

#: S2 -- vertical growth anisotropy: d(height)/dt divided by d(width)/dt. The denominator
#: is floored so a component that does not widen at all does not produce an infinity.
S2_WIDTH_RATE_FLOOR_PX_PER_FRAME: Final[float] = 0.25

#: S3 -- diffuse boundary contrast. Boundary band width as a fraction of min(w, h),
#: floored at 3 px. FIXED A PRIORI AND NEVER FITTED: this is the constant the Sep 1 gate
#: was run with, and refitting it after the gate would retro-actively invalidate the gate.
S3_BAND_FRACTION: Final[float] = 0.15
S3_BAND_MIN_PX: Final[int] = 3

#: S4 -- DIS optical-flow drift, computed INSIDE the component bounding box only.
#: Under ablation gate: if S4 does not improve DEV median TTD it is deleted, and
#: cv2.DISOpticalFlow_create goes with it (defended surfaces 23 -> 22).
S4_ENABLED: Final[bool] = True
S4_MIN_BBOX_PX: Final[int] = 24

EPS: Final[float] = 1e-8

# ---------------------------------------------------------------------------------
# rank.py -- fusion weights: fit on DEV, deliberately interpretable
# ---------------------------------------------------------------------------------

#: A weighted logistic on 3-4 features, not a black box, because the UI has to show its
#: parts: every ranked row displays its own S1/S2/S3 chips. A gradient-boosted ensemble
#: would score the same and be unshowable.
#:
#: hazard = sigmoid(BIAS + w1*S1 + w2*S2 + w3*S3 + w4*S4 + w_esc*escalation_score)
FUSION_BIAS: Final[float] = -3.2
FUSION_W_S1: Final[float] = 1.9
FUSION_W_S2: Final[float] = 1.4
FUSION_W_S3: Final[float] = 2.6   # the fog discriminator carries the most weight
FUSION_W_S4: Final[float] = 0.8
FUSION_W_ESCALATION: Final[float] = 1.5

#: The watch list a human actually reads. Detection means "entered the top-k AND fired an
#: alert" -- being ranked 27th of 40 is not a detection (metrics.py).
TOP_K: Final[int] = 4

#: The operating point the whole project is calibrated to. Not fitted -- chosen, because
#: it is the number fire agencies procure on.
BUDGET_ALERTS_PER_CAMERA_NIGHT: Final[float] = 2.0
BUDGET_DRIFT_TOLERANCE: Final[float] = 0.15  # the CloudWatch drift alarm's band

#: solve_threshold bisects the hazard threshold on the CONFUSE arm until the realised
#: rate hits the target. These bound and terminate the bisection; they are not fitted.
THRESHOLD_SEARCH_LO: Final[float] = 0.0
THRESHOLD_SEARCH_HI: Final[float] = 1.0
THRESHOLD_SEARCH_ITERS: Final[int] = 60
THRESHOLD_SEARCH_TOL: Final[float] = 1e-4

#: Once a camera has spent its two alerts for the night, further hazard rises still show
#: the row -- the system does not silently keep alerting, and it does not silently stop
#: showing either (ui.md section 7, "budget exhausted").
ALERTS_PER_CAMERA_PER_NIGHT: Final[int] = 2

# ---------------------------------------------------------------------------------
# escalate.py -- the ~1% path
# ---------------------------------------------------------------------------------

TILE_SIZE: Final[int] = 96
ESCALATION_TOP_K: Final[int] = 2          # only the top-2 cameras are ever escalated
ESCALATION_HAZARD_MIN: Final[float] = 0.35
#: Asserted by a test on every run: the ONNX must touch at most 2% of frames, or the
#: 99/1 cost architecture is a claim rather than a fact.
ESCALATION_FRACTION_MAX: Final[float] = 0.02

#: B1, the headline baseline: the SAME onnx over a full-frame tile grid, max score.
B1_GRID_STRIDE: Final[int] = 48           # 50% overlap at TILE_SIZE=96

# ---------------------------------------------------------------------------------
# telemetry.py
# ---------------------------------------------------------------------------------

CLOUDWATCH_NAMESPACE: Final[str] = "PlumeRank"
HEARTBEAT_SECONDS: Final[float] = 60.0
#: The replay loop starts on the first SSE client and stops this long after the last one
#: disconnects. t4g.small is burstable: a wall that ran continuously would sit above the
#: 20%-per-vCPU baseline all day and throttle the endpoint mid-judging
#: (architecture.md section 3, complexity.md A13).
REPLAY_IDLE_STOP_SECONDS: Final[float] = 120.0

# ---------------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------------

_SOURCE = pathlib.Path(__file__)


def freeze_hash() -> str:
    """sha256 of this file's bytes -- what ``results/constants.sha256`` records.

    Hashing the source rather than a dict of values is deliberate: it catches a changed
    comment that reveals a changed intent, and it cannot be spoofed by a constant that
    is computed at import time.
    """
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover - the freeze is run by hand, once
    print(freeze_hash())
