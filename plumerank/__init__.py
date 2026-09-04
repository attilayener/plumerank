"""PlumeRank -- attention allocation for wildfire camera networks.

The one sentence this package implements:

    A camera network is replayed minute by minute; for every minute PlumeRank
    produces a ranked watch list calibrated to 2 alerts per camera-night; and the
    harness measures how many minutes after ignition the true camera first enters
    the top-k.

Module map (seven, and no eighth -- architecture.md section 7):

    constants.py   every fitted number in the project, frozen and hashable
    types.py       the dataclasses the stages hand each other
    metrics.py     time-to-detection against the ignition clock; p50/p95; TickMeter
    stabilise.py   phaseCorrelate + estimateAffine2D(USAC_MAGSAC) fallback -- motion removal
    segment.py     CLAHE -> MOG2 / LSBP / GSOC -> morphology -> connected components
    statistic.py   THE PRIMITIVE: S1 anchoring, S2 anisotropy, S3 diffuse boundary, S4 drift
    rank.py        hazard fusion -> budget threshold -> ordered watch list -> evidence JPEG
    escalate.py    the ONNX tile classifier through cv2.dnn, on ~1% of frames
    telemetry.py   CloudWatch PutMetricData -- the only boto3 in the project
"""

__version__ = "0.1.0"
