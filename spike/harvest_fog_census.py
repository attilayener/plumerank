"""Draw an UNBIASED pool of dawn frames, so the fog class can be rebuilt from labels.

Why this script exists
----------------------
``results/fog_class_review.md`` found that roughly half of ``spike/urls_fog.txt`` -- the 60
frames the whole ``fog_marine`` class is calibrated on -- shows no marine layer at all. The
class was selected by a within-camera fog score and then, when the a-priori predicate
admitted zero of the 60, the predicate was **re-derived from those same 60 frames**
(``gonogo_2026-09-01.md`` section 7.5). A rule fitted to a contaminated set inherits the
contamination, and it did: the CONFUSE arm's ``fog_marine`` windows are chosen by it.

``fog_class_review.md`` section 5 is explicit that the fix is **new frames, not a new rule over
the old ones**, and section 6 orders the work: a human re-walks the old 60 first, and only then
is a class rebuilt. This script prepares BOTH halves of that hour so it is one sitting
rather than two.

What it produces, and why it is shaped this way
-----------------------------------------------
**1. An unbiased sampling frame, not an enriched one.** Every second morning of the season
on all six calibration cameras at one local time -- no fog score, no screening, no
predicate. Screening is what produced the current class, and screening a pool before a
human labels it is the same circularity in a new coat. The cost is that most frames will
not be fog; the benefit is that the labels which come back are a *measurement*, and they
carry something the project has never had: **the base rate of marine layer on these
cameras at this hour**. A predicate's precision cannot be read without it.

**2. The old 60 are shuffled in, and the labelling is BLIND.** ``fog_class_review.md``
section 3 flags its own method as "a visual call made ... in one pass, knowing that every
frame was labelled fog". That is the objection, and it is a fair one in both directions --
knowing the label can manufacture agreement as easily as disagreement. So the old 60 and
the fresh pool are interleaved under a fixed seed and the labelling tool shows an index
and nothing else. Whichever way the census comes out, it comes out of a procedure that
could not have known which frame was which.

**3. Nothing here decides anything.** The output is a pool and a tool. No frame is
promoted to ``fog_marine``, no cutline moves, no manifest is rewritten. Those follow the
labels, in ``scripts/build_confusion_set.py``, where they already live.

Where this lives, and why not in ``scripts/``
---------------------------------------------
``specs/build-contract.md`` W6 fixes ``scripts/`` at six files. This is a one-off
measurement that answers one question and is then finished, which is what ``spike/``
already holds -- ``propose_boxes.py`` and ``s3_separation.py`` are the same kind of
object, and ``spike/`` is judge-facing, so the census is auditable without becoming part
of the pipeline's surface.

Reproduce::

    python3 spike/harvest_fog_census.py            # ~15 min, ~300 GETs
    open spike/census.html                         # label, then Save
    python3 spike/harvest_fog_census.py --report   # the census, from the labels

Imagery: HPWREN, http://hpwren.ucsd.edu/ -- fetched at the same six-socket politeness
budget as the rest of this project, and never committed (``spike/frames/`` is ignored).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The archive walk, the local-time quarter rule and the politeness budget already exist in
# build_confusion_set.py, which owns the fog class this census is about. Importing them
# rather than copying is what makes the census comparable to the selection it audits: the
# same URL rule, the same six sockets, the same retry policy that refuses to read a flaky
# network as an empty archive.
from build_confusion_set import (  # noqa: E402
    MAX_PROBE_CONCURRENCY,
    PACIFIC,
    PHASE_SEARCH_SECONDS,
    frame_url,
    probe_exists,
)
from fetch import FetchError, fetch_with_retry  # noqa: E402

# ---------------------------------------------------------------------------------
# The sampling frame -- every number here is a decision about what the census can see
# ---------------------------------------------------------------------------------

#: The six cameras that contributed the current fog class. Holding the camera set fixed is
#: what makes the new census comparable to the old one: if the pool moved to other
#: cameras, a difference in the fog rate could be the cameras rather than the labelling.
#: It also keeps ``test_confuse_shares_cameras`` satisfiable -- every CONFUSE window must
#: sit on a camera that appears in the HOLD roster.
CENSUS_CAMERAS: tuple[str, ...] = (
    "hp-s-mobo-c",
    "hp-e-mobo-c",
    "hp-w-mobo-c",
    "wilson-s-mobo-c",
    "wilson-e-mobo-c",
    "tdlln-n-mobo-c",
)

#: The season the current fog class was drawn from (2026-06-02 to 2026-08-28), extended to
#: the last morning the archive can be expected to hold. Same season, same cameras: the
#: only thing this script changes about the old selection is the *rule*.
CENSUS_START: str = "2026-06-01"
CENSUS_END: str = "2026-09-05"

#: Consecutive mornings are not independent -- the same marine layer sits over the same
#: ridge for a week -- so the sweep strides. ``build_confusion_set`` uses 3 for the same
#: reason; 2 here because the pool is the measurement rather than a screen, and a coarser
#: stride would buy politeness at the cost of the base rate's precision.
CENSUS_DAY_STRIDE: int = 2

#: One frame per camera-morning, at the middle of the 06:00-08:59 local band the current
#: fog class was drawn from. One frame and not three: within a morning, frames 70 minutes
#: apart are the same weather, and three of them would triple the human's labelling time
#: while adding almost nothing to the census.
CENSUS_LOCAL_HOUR: int = 7
CENSUS_LOCAL_MINUTE: int = 30

#: Long side of the labelling copy. Full frames are 3072x2048; a marine layer is a
#: whole-frame feature and is unmistakable at this width, and 350 full frames is 175 MB of
#: somebody else's bandwidth held on disk for no gain. The URL is kept in the pool, so any
#: frame the statistics later need can be re-fetched at full resolution.
LABEL_WIDTH: int = 1024

#: The interleave. Fixed so the pool is reproducible, and recorded in the pool file so the
#: blinding can be audited after the fact rather than trusted.
SHUFFLE_SEED: int = 20260906

POOL_PATH = ROOT / "spike" / "fog_census_pool.json"
MANIFEST_PATH = ROOT / "spike" / "census_manifest.js"
LABELS_PATH = ROOT / "spike" / "fog_census_labels.json"
FRAME_DIR = ROOT / "spike" / "frames" / "census"
LEGACY_FRAME_DIR = ROOT / "spike" / "frames"
FOG_URL_LIST = ROOT / "spike" / "urls_fog.txt"

#: The three answers. ``overcast`` exists because it is the category
#: ``fog_class_review.md`` section 3 could not call either way -- a stratus deck that has closed
#: over the camera IS the marine layer to a meteorologist, and is not a fog BANK with a
#: visible leading edge, which is the only thing S3 can be asked about. Collapsing it into
#: either neighbour would hide the disagreement that matters.
LABELS: tuple[str, ...] = ("marine_layer", "overcast", "neither")


# ---------------------------------------------------------------------------------
# Walking the archive
# ---------------------------------------------------------------------------------


def label_filename(camera: str, epoch: int) -> str:
    """``<camera>_<epoch>.jpg`` -- and the camera is not decoration.

    Naming a labelling copy by its epoch alone silently loses frames. Cameras on one mast
    write on the same 60 s grid at the same phase: ``hp-s-mobo-c`` and ``hp-w-mobo-c`` landed
    on the *identical second* on 16 of the 49 sampled mornings, so 16 pairs collided, one
    overwrote the other, and 16 pool rows pointed at another camera's picture. A labeller
    would have seen the same image twice and labelled it for two different cameras.

    Caught by counting unique filenames against pool rows -- 309 against 325 -- which is now
    an assertion in ``build_pool`` rather than something someone has to think to check.
    """
    return f"{camera}_{int(epoch)}.jpg"


def census_days() -> list[str]:
    """Every sampled morning, as ``YYYY-MM-DD``, oldest first."""
    start = datetime.strptime(CENSUS_START, "%Y-%m-%d").date()
    end = datetime.strptime(CENSUS_END, "%Y-%m-%d").date()
    days: list[str] = []
    day = start
    while day <= end:
        days.append(day.isoformat())
        day += timedelta(days=CENSUS_DAY_STRIDE)
    return days


def target_epoch(day: str, hour: int, minute: int) -> int:
    """The epoch of ``hour:minute`` San Diego local on ``day``.

    Local and not UTC, and for the same reason the archive's own directories are local:
    what is being matched is the sun angle. A UTC clock would drift an hour across the DST
    boundary and quietly sample a different part of dawn on either side of it.
    """
    naive = datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, minute=minute)
    return int(naive.replace(tzinfo=PACIFIC).timestamp())


def find_frame(camera: str, day: str) -> int | None:
    """The epoch of the frame nearest the target minute on this camera-morning, or None.

    A camera writes on a 60 s grid at its own phase, so the target second is almost never
    a key. The search walks outward from the target -- 0, -1, +1, -2, +2 -- across a whole
    minute, which is guaranteed to contain exactly one frame if the camera was writing at
    all. A morning with no frame within the minute is a camera that was down, and it is
    recorded as a hole rather than back-filled from a neighbouring hour: a frame from
    08:40 is a different sun angle, and the census is a statement about one.
    """
    base = target_epoch(day, CENSUS_LOCAL_HOUR, CENSUS_LOCAL_MINUTE)
    for step in range(PHASE_SEARCH_SECONDS + 1):
        for delta in ((0,) if step == 0 else (-step, step)):
            if probe_exists(frame_url(camera, base + delta)):
                return base + delta
    return None


def _label_copy(payload: bytes, destination: pathlib.Path) -> tuple[int, int]:
    """Decode, downscale to ``LABEL_WIDTH``, write. Returns the source frame's shape."""
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FetchError(f"undecodable frame for {destination.name}")
    height, width = image.shape[:2]
    scale = LABEL_WIDTH / width
    small = cv2.resize(
        image,
        (LABEL_WIDTH, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return width, height


#: Camera-mornings already resolved by a previous run, ``(camera, date) -> epoch``. A
#: re-run must not re-probe: discovering one morning's phase costs up to 61 HEADs against a
#: public academic archive, and the answer cannot change once found -- the frame either
#: exists at that second or the archive has been rewritten. Populated from the committed
#: pool file, so the cache survives a clean checkout and is auditable rather than hidden in
#: a temp directory. Only mornings whose image is still on disk are reused; a deleted frame
#: falls back to the probe.
_KNOWN: dict[tuple[str, str], int] = {}
#: Source frame shape per resolved camera-morning, from the committed pool -- see harvest_one.
_KNOWN_SHAPE: dict[tuple[str, str], tuple[int, int]] = {}

#: Camera-mornings a previous run searched and found empty. Cached for the same reason as
#: the hits, and it is the more expensive half: a hole costs the FULL 61-probe search,
#: where a hit usually stops early. ``probe_exists`` raises on transport errors rather than
#: reading them as absence, so a cached hole is a real 403 and not a flaky socket. Delete
#: spike/fog_census_pool.json to search the whole grid again.
_KNOWN_HOLES: set[tuple[str, str]] = set()


def _load_known() -> None:
    if not POOL_PATH.exists():
        return
    payload = json.loads(POOL_PATH.read_text())
    for entry in payload["entries"]:
        if entry.get("source") == "current_fog_class":
            continue
        if (FRAME_DIR / label_filename(entry["camera"], entry["epoch"])).exists():
            _KNOWN[(entry["camera"], entry["date"])] = entry["epoch"]
            _KNOWN_SHAPE[(entry["camera"], entry["date"])] = (
                entry["source_width"],
                entry["source_height"],
            )
    for hole in payload["sampling_frame"]["holes"]:
        _KNOWN_HOLES.add((hole["camera"], hole["date"]))


def harvest_one(camera: str, day: str) -> dict | None:
    """One camera-morning: find the frame, fetch it, write the labelling copy."""
    if (camera, day) in _KNOWN_HOLES:
        return None
    epoch = _KNOWN.get((camera, day)) or find_frame(camera, day)
    if epoch is None:
        return None
    url = frame_url(camera, epoch)
    destination = FRAME_DIR / label_filename(camera, epoch)
    if not destination.exists():
        width, height = _label_copy(fetch_with_retry(url), destination)
    else:
        # The file on disk is the LABEL copy (1024 wide), not the source. The source shape
        # was recorded when it was fetched; carry it forward rather than re-reading the
        # copy as if it were the archive frame -- the 2026-09-16 re-run did exactly that
        # and wrote 1024x683 over 63 rows that had said 3072x2048.
        width, height = _KNOWN_SHAPE.get((camera, day), (0, 0))
    return {
        "camera": camera,
        "date": day,
        "epoch": epoch,
        "url": url,
        "file": f"census/{label_filename(camera, epoch)}",
        "source": "fresh",
        "source_width": width,
        "source_height": height,
    }


def legacy_entries() -> list[dict]:
    """The current fog class, as pool rows, from frames already on disk.

    These are the 60 in ``spike/urls_fog.txt``. They are already fetched at full resolution
    for the S3 gate, so nothing is re-fetched -- but they are **re-encoded to the census
    width**, and that is not housekeeping. Served as they sit, they would be 3072x2048 among
    325 frames that are 1024 wide, and a labeller who noticed one frame looking sharper than
    the last would have been handed exactly the fact the blinding exists to withhold. Every
    frame in the pool goes through the same resize and the same JPEG quality.
    """
    rows: list[dict] = []
    for url in FOG_URL_LIST.read_text().split():
        name = url.rsplit("/", 1)[-1]
        epoch = int(name.split(".")[0])
        camera = url.split("/MTA/", 1)[1].split("/", 1)[0]
        date = datetime.fromtimestamp(epoch, PACIFIC).date().isoformat()
        destination = FRAME_DIR / label_filename(camera, epoch)
        source = LEGACY_FRAME_DIR / name
        if not destination.exists():
            if not source.exists():
                raise FetchError(
                    f"{source} is missing -- the S3 gate frames are fetched by "
                    f"spike/s3_separation.py from spike/urls.txt"
                )
            _label_copy(source.read_bytes(), destination)
        rows.append(
            {
                "camera": camera,
                "date": date,
                "epoch": epoch,
                "url": url,
                "file": f"census/{label_filename(camera, epoch)}",
                "source": "current_fog_class",
                "source_width": 3072,
                "source_height": 2048,
            }
        )
    return rows


# ---------------------------------------------------------------------------------
# Building the pool
# ---------------------------------------------------------------------------------


def build_pool() -> dict:
    """Sweep the archive, interleave the old class in, write the pool and the manifest."""
    _load_known()
    days = census_days()
    jobs = [(camera, day) for day in days for camera in CENSUS_CAMERAS]
    if _KNOWN or _KNOWN_HOLES:
        print(
            f"reusing {len(_KNOWN)} camera-mornings already resolved and "
            f"{len(_KNOWN_HOLES)} known holes -- no re-probing"
        )
    print(
        f"sweeping {len(CENSUS_CAMERAS)} cameras x {len(days)} mornings "
        f"= {len(jobs)} camera-mornings at "
        f"{CENSUS_LOCAL_HOUR:02d}:{CENSUS_LOCAL_MINUTE:02d} local",
        flush=True,
    )

    fresh: list[dict] = []
    holes: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_PROBE_CONCURRENCY) as pool:
        for (camera, day), row in zip(
            jobs, pool.map(lambda job: _safe_harvest(*job), jobs)
        ):
            if row is None:
                holes.append({"camera": camera, "date": day})
            else:
                fresh.append(row)
            done = len(fresh) + len(holes)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}  found {len(fresh)}", flush=True)

    # A morning the old class already sampled can also fall on the census grid -- the
    # class's middle frame sits at 07:29 local, which is the census minute. Showing the
    # same picture twice would double-count it and would let a labeller notice the repeat,
    # which is a hole in the blinding. One row per epoch, marked as belonging to both
    # arms, and the report counts it in both.
    legacy = legacy_entries()
    legacy_epochs = {row["epoch"] for row in legacy}
    overlap = sorted(row["epoch"] for row in fresh if row["epoch"] in legacy_epochs)
    fresh_only = [row for row in fresh if row["epoch"] not in legacy_epochs]
    for row in legacy:
        if row["epoch"] in set(overlap):
            row["source"] = "both"

    entries = fresh_only + legacy
    files = {entry["file"] for entry in entries}
    if len(files) != len(entries):
        raise SystemExit(
            f"{len(entries) - len(files)} pool rows share a labelling file. Two rows pointing "
            f"at one picture means one frame was overwritten by another and the labeller will "
            f"never see it -- see label_filename()."
        )
    random.Random(SHUFFLE_SEED).shuffle(entries)
    for index, entry in enumerate(entries):
        entry["index"] = index

    pool_payload = {
        "generated": datetime.now(PACIFIC).isoformat(timespec="seconds"),
        "why": (
            "An unbiased dawn pool for re-deciding what the fog class is. No fog score, "
            "no predicate, no screening -- screening a pool before a human labels it is "
            "the circularity that produced the current class."
        ),
        "sampling_frame": {
            "cameras": list(CENSUS_CAMERAS),
            "start": CENSUS_START,
            "end": CENSUS_END,
            "day_stride": CENSUS_DAY_STRIDE,
            "local_time": f"{CENSUS_LOCAL_HOUR:02d}:{CENSUS_LOCAL_MINUTE:02d}",
            "frames_per_camera_morning": 1,
            "camera_mornings_attempted": len(jobs),
            "camera_mornings_found": len(fresh),
            "holes": holes,
            "overlap_with_current_fog_class": overlap,
        },
        "blinding": {
            "method": (
                "The 60 frames of the current fog class are interleaved with the fresh "
                "pool under a fixed seed and the labelling tool shows an index only. The "
                "labeller cannot tell which frames were previously called fog. Every frame "
                "is served at the same width and JPEG quality, so the two arms cannot be "
                "told apart by how the picture looks either."
            ),
            "seed": SHUFFLE_SEED,
            "known_limits": (
                "The archive burns a banner into the top of every frame carrying the camera "
                "name and the exact timestamp. spike/census.html hides it with a display "
                "crop of the top 2.7%, and the files on disk are untouched -- so the "
                "blinding holds for anyone using the tool as intended, and someone who "
                "opened the JPEGs directly could still read a date. Recorded rather than "
                "claimed away."
            ),
            "current_fog_class_frames": len(legacy),
            "fresh_frames": len(fresh),
            "frames_in_both_arms": len(overlap),
        },
        "labels": list(LABELS),
        "credit": "Imagery: HPWREN, http://hpwren.ucsd.edu/",
        "entries": entries,
    }
    POOL_PATH.write_text(json.dumps(pool_payload, indent=1) + "\n")

    manifest = [
        {"i": e["index"], "file": e["file"]} for e in sorted(entries, key=lambda e: e["index"])
    ]
    MANIFEST_PATH.write_text(
        "// Generated by spike/harvest_fog_census.py -- do not hand-edit.\n"
        "// Index and filename only: the labelling pass is BLIND to camera, date and\n"
        "// to whether a frame belongs to the current fog class.\n"
        "const CENSUS = " + json.dumps(manifest, indent=1) + ";\n"
    )
    print(
        f"\npool: {len(entries)} frames "
        f"({len(fresh)} fresh + {len(legacy)} current class, "
        f"{len(overlap)} in both), "
        f"{len(holes)} camera-mornings with no frame\n"
        f"wrote {POOL_PATH.relative_to(ROOT)} and {MANIFEST_PATH.relative_to(ROOT)}\n"
        f"next: open spike/census.html, label, Save to "
        f"{LABELS_PATH.relative_to(ROOT)}"
    )
    return pool_payload


def _safe_harvest(camera: str, day: str) -> dict | None:
    """``harvest_one`` with a missing frame treated as a hole, not as a crash.

    A transport failure still raises -- ``probe_exists`` and ``fetch_with_retry`` both
    refuse to read a flaky network as an empty archive, and that refusal is the whole
    reason the census can be quoted as a base rate.
    """
    try:
        return harvest_one(camera, day)
    except FetchError as exc:
        print(f"  ! {camera} {day}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------------
# The census, once the labels come back
# ---------------------------------------------------------------------------------


def report() -> int:
    """Read the labels, print the census, and say what it means for the current class."""
    if not LABELS_PATH.exists():
        print(f"no labels yet -- open spike/census.html, label, then Save to {LABELS_PATH}")
        return 1
    pool = json.loads(POOL_PATH.read_text())
    labels = json.loads(LABELS_PATH.read_text())
    by_index = {int(k): v for k, v in labels["labels"].items()}
    entries = {e["index"]: e for e in pool["entries"]}

    counts: dict[str, collections.Counter] = {
        "fresh": collections.Counter(),
        "current_fog_class": collections.Counter(),
    }
    #: A frame in both arms is one picture with one label, counted once in each arm --
    #: it is genuinely a member of both the sampling frame and the current class.
    arms = {
        "fresh": ("fresh", "both"),
        "current_fog_class": ("current_fog_class", "both"),
    }
    for index, label in by_index.items():
        source = entries[index]["source"]
        for arm, members in arms.items():
            if source in members:
                counts[arm][label] += 1

    print("\nThe census, blind\n" + "=" * 60)
    header = f"{'':22}" + "".join(f"{name:>15}" for name in LABELS) + f"{'total':>9}"
    print(header)
    for source, counter in counts.items():
        total = sum(counter.values())
        row = f"{source:22}" + "".join(f"{counter[name]:>15}" for name in LABELS)
        print(row + f"{total:>9}")

    fresh_total = sum(counts["fresh"].values())
    if fresh_total:
        rate = counts["fresh"]["marine_layer"] / fresh_total
        print(
            f"\nBase rate of marine layer on these six cameras at "
            f"{CENSUS_LOCAL_HOUR:02d}:{CENSUS_LOCAL_MINUTE:02d} local, "
            f"{CENSUS_START} to {CENSUS_END}: {rate:.1%} "
            f"({counts['fresh']['marine_layer']}/{fresh_total})"
        )
    old_total = sum(counts["current_fog_class"].values())
    if old_total:
        old_rate = counts["current_fog_class"]["marine_layer"] / old_total
        print(
            f"Of the CURRENT fog class, marine layer: {old_rate:.1%} "
            f"({counts['current_fog_class']['marine_layer']}/{old_total})"
        )
        if fresh_total:
            print(
                "\nThe current class was selected to be fog. If its marine-layer rate is "
                "not far above the base rate, the selection did not work, and "
                "fog_class_review.md's census is confirmed by a blind pass."
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="read spike/fog_census_labels.json and print the census",
    )
    args = parser.parse_args(argv)
    if args.report:
        return report()
    build_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
