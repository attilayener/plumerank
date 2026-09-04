"""Pull exactly the frames the committed manifests name, and prove they did not change.

Two files, two different jobs, and confusing them is how a corpus quietly rots:

* **The manifest pins WHICH frames.** ``data/manifest_{dev,hold,confuse}.json`` are
  committed lists of sequence ids and frame names. They are the experiment's design.
* **The receipt pins WHAT they contained.** On the first fetch this script writes
  ``data/fetch_receipt.json`` with a sha256 per file; every later run re-hashes what is on
  disk, compares against that receipt, and exits non-zero on any disagreement.

Neither file alone is enough. A manifest without a receipt lets HPWREN re-encode a frame
under the same name and silently move the published headline; a receipt without a manifest
cannot tell a missing frame from a frame that was never wanted. Both are committed, so a
judge who clones this repo and runs ``make fetch`` either gets the same images this project
was measured on or gets a non-zero exit saying which one differs. The count is whatever the
committed manifests add up to -- 2,409 DEV + 4,827 HOLD + 3,645 CONFUSE = 10,881 today --
and it is printed per manifest at the end of the run rather than asserted here, because a
number in a docstring cannot be re-derived and this one was wrong once already.

Only the named windows -- never the archive
-------------------------------------------
FIgLib publishes 521 sequences. This script fetches 90 of them plus the 45 CONFUSE windows,
because that is what the manifests name. Pulling the archive "to be safe" would be ~40 GB
of somebody else's bandwidth for frames no arm of the evaluation ever reads.

FETCHING IS NOT READING
-----------------------
Writing HOLD bytes to disk is not looking at them, and this script therefore does **not**
append to ``results/hold_access.log``. Only ``scripts/bench.py``, which puts HOLD pixels
through the detector, does. That distinction is what makes the log a meaningful record
instead of a line that fires whenever anyone runs ``make fetch``; ``tests/test_fetch.py``
asserts both the behaviour and the absence of the string from this source.

One engine, no per-corpus code
------------------------------
Every manifest -- FIgLib windows, MTA confusion windows, the Pyro-SDIS training mirror --
is read through the same four keys (``base_url``, ``windows[].sequence_id``,
``windows[].frames``, and an optional per-window ``url_prefix``). A corpus that needs a
branch in here is a corpus whose manifest is underspecified, and the fix belongs in the
manifest.

Usage::

    python scripts/fetch.py --corpus figlib     # DEV + HOLD + CONFUSE, then build micro
    python scripts/fetch.py --corpus pyrosdis   # the tile classifier's training mirror
    python scripts/fetch.py --corpus all        # both; what `make fetch` runs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence
from urllib.parse import quote

import cv2

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root, so the package
# this script exists to serve would otherwise be unimportable from its own repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plumerank.constants import WORK_HEIGHT, WORK_WIDTH  # noqa: E402

__all__ = [
    "FetchError",
    "PlannedFile",
    "frame_url",
    "http_get",
    "fetch_with_retry",
    "sha256_bytes",
    "sha256_file",
    "load_manifest",
    "plan_manifest",
    "fetch_planned",
    "build_micro",
    "main",
]

# ---------------------------------------------------------------------------------
# Being a polite client
# ---------------------------------------------------------------------------------

#: HPWREN gives this imagery away and asks for a credit reference in return. The least a
#: client can do is say who it is, so an operator reading an access log can find us
#: instead of blackholing an anonymous burst of requests.
USER_AGENT: str = (
    "PlumeRank/0.1 (OpenCV AI Competition 2026 entry; fetches only the frames named in "
    "its committed manifests; contact via the repository README)"
)

#: Eight sockets, and the cap is enforced rather than documented. The archive is a CDN in
#: front of one research group's storage; the marginal frame arrives no faster past this
#: point and the marginal request costs them the same either way.
MAX_CONCURRENCY: int = 8

HTTP_TIMEOUT_SECONDS: float = 30.0
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 1.5

#: Five DEV sequences, about 400 frames -- 402 today, because three of the five windows the
#: selection rule picks are 80 frames rather than 81: the archive is missing one minute in
#: each and the manifest records that rather than papering over it. This is the whole reason
#: `make verify` finishes in three minutes with no network and no credentials.
MICRO_SEQUENCES: int = 5

#: The micro frames are committed, so this quality setting is baked into bytes that are in
#: the repository -- it is read once, by whoever regenerates them, and never by the judge
#: path. High enough that the downscale, not the encoder, is what the pipeline sees.
MICRO_JPEG_QUALITY: int = 92

#: The fetch group named by ``--corpus``. The spelling is fixed by the Makefile and the
#: build contract; the grouping is by *what a run needs*, not by which archive the bytes
#: came from -- ``figlib`` is every HPWREN camera window the evaluation replays, and the
#: CONFUSE windows are HPWREN MTA rather than FIgLib.
_HPWREN_MANIFESTS: tuple[str, ...] = (
    "manifest_dev.json",
    "manifest_hold.json",
    "manifest_confuse.json",
)
_MANIFESTS_BY_CORPUS: dict[str, tuple[str, ...]] = {
    "figlib": _HPWREN_MANIFESTS,
    "pyrosdis": ("manifest_pyrosdis.json",),
    "all": _HPWREN_MANIFESTS + ("manifest_pyrosdis.json",),
}

#: Where an arm's bytes land under ``--data-root``. A manifest may override with its own
#: ``local_root`` key; this table only covers the arms whose manifests were committed
#: before the key existed.
_LOCAL_ROOT_BY_ARM: dict[str, str] = {
    "dev": "corpus/figlib",
    "hold": "corpus/figlib",
    "confuse": "corpus/mta",
    "pyrosdis": "pyrosdis",
}

RECEIPT_NAME: str = "fetch_receipt.json"


class FetchError(RuntimeError):
    """A fetch that cannot be completed correctly. Never caught to produce a default."""


# ---------------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------------


def frame_url(prefix: str, frame: str) -> str:
    """``prefix`` joined to one frame name, percent-encoded the way the archive needs.

    FIgLib names the post-ignition frames ``1719093958_+00060.jpg``. A literal ``+`` in a
    URL path is legal but is read as a space by enough intermediaries that the CDN returns
    the wrong key for some of them, so it is encoded to ``%2B``: ``quote`` does that,
    because ``+`` is not in its unreserved set.

    ``safe="/"`` and not ``safe=""``: MTA frame names carry their quarter directory
    (``large/20260726/Q3/1785071971.jpg``) so that one CONFUSE window can span the two
    quarter directories an 81-minute span falls across. FIgLib names contain no slash, so
    the same call is correct for both and neither corpus needs a branch.
    """
    return prefix + quote(frame, safe="/")


def http_get(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> bytes:
    """One GET. The single seam the tests replace -- nothing else in here opens a socket."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch_with_retry(url: str, attempts: int = RETRY_ATTEMPTS) -> bytes:
    """GET with exponential backoff on *transient* failures only.

    A 404 or a 403 is not retried. The archive is S3-backed with listing disabled, so a
    missing key answers 403 rather than 404 -- both mean "this frame does not exist", and
    asking three more times neither produces it nor is polite. Retrying is reserved for
    5xx and transport errors, which are the ones a second attempt can actually fix.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            # Resolved through the module global on every call so that the one seam the
            # tests replace stays one seam, rather than a callable threaded through five
            # signatures that production never passes.
            return http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise FetchError(f"{url} -> HTTP {exc.code}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < attempts:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    raise FetchError(f"{url} failed after {attempts} attempts: {last}")


# ---------------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """One frame: where it comes from and where it goes. Nothing corpus-specific."""

    url: str
    relative_path: str
    """POSIX, relative to ``--data-root``. It is the receipt's key, so it must not carry a
    platform separator -- a receipt written on Windows would otherwise never match one
    written on Linux and the mismatch would look like corruption."""


def load_manifest(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise FetchError(
            f"{path} is missing. This script fetches what a committed manifest names and "
            f"never guesses a URL scheme, so an absent manifest is a hard stop rather "
            f"than an empty fetch that exits zero."
        )
    return json.loads(path.read_text())


def plan_manifest(manifest: dict, manifest_path: pathlib.Path) -> list[PlannedFile]:
    """Every file one manifest names, expanded once, before a single byte is requested.

    Planning ahead rather than streaming means the run knows its own size, the concurrency
    pool sees one flat work list instead of a nested loop, and a manifest that is malformed
    fails before it has half-populated a corpus directory.
    """
    arm = manifest.get("arm")
    base_url = manifest.get("base_url", "")
    local_root = manifest.get("local_root") or _LOCAL_ROOT_BY_ARM.get(str(arm))
    if local_root is None:
        raise FetchError(
            f"{manifest_path}: arm {arm!r} has no local_root, and this script has no "
            f"default for it. Add a 'local_root' key to the manifest."
        )

    planned: list[PlannedFile] = []
    for position, window in enumerate(manifest.get("windows", [])):
        try:
            sequence_id = window["sequence_id"]
            frames = window["frames"]
        except KeyError as exc:
            raise FetchError(
                f"{manifest_path}: window {position} has no {exc.args[0]!r}. A manifest "
                f"this script cannot read is a corpus it must not half-fetch."
            ) from exc
        prefix = window.get("url_prefix") or (base_url + sequence_id + "/")
        declared = window.get("n_frames")
        if declared is not None and declared != len(frames):
            raise FetchError(
                f"{manifest_path}: window {sequence_id} declares n_frames={declared} but "
                f"lists {len(frames)} frames -- the design and the list disagree."
            )
        for frame in frames:
            planned.append(
                PlannedFile(
                    url=frame_url(prefix, frame),
                    relative_path=f"{local_root}/{sequence_id}/{frame}",
                )
            )
    return planned


# ---------------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------------


def read_receipt(path: pathlib.Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("files", {})


def write_receipt(path: pathlib.Path, files: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": date.today().isoformat(),
        "note": (
            "The manifests pin WHICH frames; this receipt pins WHAT they contained. "
            "scripts/fetch.py re-hashes every file against it on every later run and "
            "exits non-zero on a mismatch."
        ),
        "n_files": len(files),
        "files": {key: files[key] for key in sorted(files)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _looks_complete(relative_path: str, data: bytes) -> bool:
    """Is a file that predates the receipt safe to adopt without re-downloading?

    Only asked on the first fetch, where there is no recorded hash to compare against and
    size alone cannot tell a finished download from an interrupted one. A JPEG carries its
    own end-of-image marker, so for the imagery -- which is all of the bytes -- the answer
    is checkable. Everything else (Pyro-SDIS label files) is adopted, because a truncated
    label file is caught by the receipt on the very next run and there is no in-band marker
    to check here.
    """
    if not relative_path.lower().endswith((".jpg", ".jpeg")):
        return True
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


# ---------------------------------------------------------------------------------
# The fetch
# ---------------------------------------------------------------------------------


@dataclass(slots=True)
class FetchResult:
    downloaded: int = 0
    skipped: int = 0
    adopted: int = 0
    mismatches: list[str] = field(default_factory=list)
    hashes: dict[str, dict] = field(default_factory=dict)


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write through a sibling temp file and rename.

    A half-written frame at its final path is indistinguishable from a complete one on the
    next run, and the next run is exactly where resumability lives. ``os.replace`` on the
    same filesystem makes "the file exists" and "the file is whole" the same statement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _one_file(
    planned: PlannedFile,
    data_root: pathlib.Path,
    receipt: dict[str, dict],
) -> tuple[str, str, dict | None, str | None]:
    """Resolve one planned file. Returns ``(kind, relative_path, entry, mismatch)``."""
    target = data_root / planned.relative_path
    recorded = receipt.get(planned.relative_path)

    if target.is_file():
        digest = sha256_file(target)
        if recorded is not None:
            if digest != recorded.get("sha256"):
                return (
                    "mismatch",
                    planned.relative_path,
                    None,
                    f"{planned.relative_path}: on disk {digest[:12]}, receipt "
                    f"{str(recorded.get('sha256'))[:12]}",
                )
            return "skipped", planned.relative_path, recorded, None
        data = target.read_bytes()
        if _looks_complete(planned.relative_path, data):
            return (
                "adopted",
                planned.relative_path,
                {"sha256": digest, "bytes": len(data)},
                None,
            )
        target.unlink()

    data = fetch_with_retry(planned.url)
    digest = sha256_bytes(data)
    if recorded is not None and digest != recorded.get("sha256"):
        # Do NOT write these bytes. The archive served something other than what this
        # project measured, and putting it on disk would make the next run agree with it.
        return (
            "mismatch",
            planned.relative_path,
            None,
            f"{planned.relative_path}: served {digest[:12]}, receipt "
            f"{str(recorded.get('sha256'))[:12]}",
        )
    _atomic_write(target, data)
    return (
        "downloaded",
        planned.relative_path,
        {"sha256": digest, "bytes": len(data)},
        None,
    )


def fetch_planned(
    planned: Sequence[PlannedFile],
    data_root: pathlib.Path,
    receipt: dict[str, dict],
    concurrency: int = MAX_CONCURRENCY,
) -> FetchResult:
    """Fetch, verify and resume one flat work list under a bounded pool."""
    workers = max(1, min(int(concurrency), MAX_CONCURRENCY))
    result = FetchResult()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one_file, item, data_root, receipt) for item in planned
        ]
        for future in concurrent.futures.as_completed(futures):
            kind, relative_path, entry, mismatch = future.result()
            if mismatch is not None:
                result.mismatches.append(mismatch)
                continue
            if entry is not None:
                result.hashes[relative_path] = entry
            setattr(result, kind, getattr(result, kind) + 1)
    return result


# ---------------------------------------------------------------------------------
# The micro corpus -- what makes `make verify` a three-minute command
# ---------------------------------------------------------------------------------


def micro_sequences(dev_manifest: dict, n_sequences: int = MICRO_SEQUENCES) -> list[dict]:
    """The DEV windows the micro corpus is cut from, chosen by a rule and not by hand.

    Sorted by ``sequence_id`` (which begins with the date) and taking the first window per
    camera: five different cameras rather than five views of whichever camera happens to
    appear most, because the micro run exercises the per-camera ranking and five windows
    from one camera would exercise a roster of one.
    """
    chosen: list[dict] = []
    seen: set[str] = set()
    for window in sorted(dev_manifest["windows"], key=lambda w: w["sequence_id"]):
        if window["camera_id"] in seen:
            continue
        seen.add(window["camera_id"])
        chosen.append(window)
        if len(chosen) == n_sequences:
            break
    if len(chosen) < n_sequences:
        raise FetchError(
            f"DEV offers only {len(chosen)} distinct cameras, fewer than the "
            f"{n_sequences} the micro corpus needs."
        )
    return chosen


def build_micro(
    dev_manifest: dict,
    data_root: pathlib.Path,
    n_sequences: int = MICRO_SEQUENCES,
) -> dict:
    """Downscale five DEV sequences to working resolution and commit them.

    This is the only part of the corpus that lives in git, and it is why the judge's
    one-liner needs no network and no credentials. The frames are re-encoded rather than
    stored as arrays so that ``make verify`` reads them through the same ``cv2.imread``
    path the real arms use -- a .npy micro corpus would verify a path that never runs.

    ``INTER_AREA`` and not the default bilinear: 3072x2048 down to 480x270 is a 6.4x
    reduction, where bilinear samples four source pixels out of forty and aliases exactly
    the thin high-contrast structure -- ridgelines, tower guy-wires -- that the boundary
    statistic then reads as texture.
    """
    micro_root = data_root / "micro"
    micro_root.mkdir(parents=True, exist_ok=True)
    source_root = data_root / (
        dev_manifest.get("local_root") or _LOCAL_ROOT_BY_ARM["dev"]
    )

    windows: list[dict] = []
    for window in micro_sequences(dev_manifest, n_sequences):
        sequence_id = window["sequence_id"]
        out_dir = micro_root / sequence_id
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for frame in window["frames"]:
            src = source_root / sequence_id / frame
            image = cv2.imread(str(src))
            if image is None:
                raise FetchError(
                    f"{src} is missing or unreadable -- the micro corpus is cut from the "
                    f"fetched DEV frames, so fetch DEV before building it."
                )
            small = cv2.resize(
                image, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA
            )
            ok, encoded = cv2.imencode(
                ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), MICRO_JPEG_QUALITY]
            )
            if not ok:
                raise FetchError(f"cv2.imencode refused {src}")
            _atomic_write(out_dir / frame, encoded.tobytes())
            written.append(frame)

        micro_window = {key: window[key] for key in window if key != "frames"}
        micro_window["n_frames"] = len(written)
        micro_window["frames"] = written
        windows.append(micro_window)

    manifest = {
        "arm": "micro",
        "n_windows": len(windows),
        "generated": date.today().isoformat(),
        "source": "HPWREN FIgLib, downscaled",
        "credit": dev_manifest["credit"],
        "licence_note": dev_manifest["licence_note"],
        "base_url": "",
        "url_note": (
            "No URL: these frames are local and committed. They are cut from the DEV "
            "windows named below by scripts/fetch.py, at "
            f"{WORK_WIDTH}x{WORK_HEIGHT} with INTER_AREA and JPEG quality "
            f"{MICRO_JPEG_QUALITY}."
        ),
        "local_root": "micro",
        "catalogue": (
            f"{len(windows)} DEV windows on {len(windows)} distinct cameras, "
            f"{sum(len(w['frames']) for w in windows)} frames."
        ),
        "selection": [
            "The DEV windows sorted by sequence_id, first per camera, first "
            f"{n_sequences}. A rule and not a hand-picked list, so the micro corpus is "
            "reproducible from the committed DEV manifest alone.",
            "DEV and never HOLD: the micro corpus is the one part of the imagery that is "
            "committed and therefore looked at, and HOLD is read exactly once.",
        ],
        "sha256_note": (
            "These frames are committed, so git is their receipt. "
            "data/fetch_receipt.json covers the fetched arms only."
        ),
        "windows": windows,
    }
    (micro_root / "manifest_micro.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _write_provenance(micro_root, dev_manifest, manifest)
    return manifest


def _write_provenance(
    micro_root: pathlib.Path, dev_manifest: dict, micro_manifest: dict
) -> None:
    """The credit reference travels with the pixels.

    ``data/micro/`` is the only imagery in the repository, so it is the only imagery that
    reaches somebody who never read ``data/manifest_dev.json``. HPWREN asks for a credit
    reference in return for the data; carrying it in the manifest alone would put it in a
    file nobody opens.
    """
    credit = dev_manifest["credit"]
    text = "\n".join(
        [
            "# data/micro/ -- provenance",
            "",
            credit,
            "",
            dev_manifest["licence_note"],
            "",
            "## What these files are",
            "",
            f"{micro_manifest['n_windows']} HPWREN FIgLib windows "
            f"({sum(len(w['frames']) for w in micro_manifest['windows'])} frames), "
            f"downscaled from 3072x2048 to {WORK_WIDTH}x{WORK_HEIGHT} with "
            "cv2.INTER_AREA and re-encoded as JPEG at quality "
            f"{MICRO_JPEG_QUALITY} by scripts/fetch.py.",
            "",
            "They exist so that `make verify` recomputes real numbers on real pixels in "
            "three minutes with no network and no credentials. They are NOT the "
            "evaluation corpus: the published headline is measured on the 60 HOLD "
            "windows by `make fetch && make bench`, and `make verify` never claims "
            "otherwise.",
            "",
            "## Regenerating them",
            "",
            "    python scripts/fetch.py --corpus figlib",
            "",
            "The selection rule is in data/micro/manifest_micro.json under `selection`.",
            "",
        ]
    )
    (micro_root / "PROVENANCE.md").write_text(text)


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def _report(name: str, result: FetchResult, total: int) -> None:
    print(
        f"{name:<24} {total:6d} files   "
        f"{result.downloaded:6d} downloaded   "
        f"{result.skipped:6d} verified   "
        f"{result.adopted:6d} adopted   "
        f"{len(result.mismatches):4d} MISMATCH"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch exactly the frames the committed manifests name, sha256-verified "
            "against data/fetch_receipt.json and resumable."
        )
    )
    parser.add_argument("--corpus", required=True, choices=sorted(_MANIFESTS_BY_CORPUS))
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=_REPO_ROOT / "data",
        help="where manifests are read and frames are written (default: ./data)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"parallel requests, capped at {MAX_CONCURRENCY}",
    )
    args = parser.parse_args(argv)

    data_root: pathlib.Path = args.data_root
    receipt_path = data_root / RECEIPT_NAME
    receipt = read_receipt(receipt_path)

    mismatches: list[str] = []
    hashes: dict[str, dict] = dict(receipt)
    dev_manifest: dict | None = None

    # Every manifest in the group is loaded and planned BEFORE the first byte is requested.
    # Doing it lazily meant `--corpus all` could spend forty minutes pulling FIgLib, hit an
    # absent pyrosdis manifest, exit 2 without writing the receipt, and repeat that on every
    # later run -- the download was never wrong, but it was never recorded either.
    plans: list[tuple[str, dict, list[PlannedFile]]] = []
    try:
        for name in _MANIFESTS_BY_CORPUS[args.corpus]:
            manifest_path = data_root / name
            manifest = load_manifest(manifest_path)
            plans.append((name, manifest, plan_manifest(manifest, manifest_path)))
    except FetchError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 2

    for name, manifest, planned in plans:
        try:
            result = fetch_planned(
                planned, data_root, receipt, concurrency=args.concurrency
            )
        except FetchError as exc:
            # An unreachable archive is a failed fetch, not a traceback. The receipt is
            # left alone: files already on disk are re-adopted on the next attempt, and
            # recording a half-corpus as the receipt would make the gap look intentional.
            print(f"FAILED  {name}: {exc}", file=sys.stderr)
            return 2
        hashes.update(result.hashes)
        mismatches.extend(result.mismatches)
        _report(name, result, len(planned))
        if manifest.get("arm") == "dev":
            dev_manifest = manifest

    if mismatches:
        # The receipt is NOT rewritten here. Recording the hashes of a run that
        # disagreed with the receipt is how a corrupted corpus becomes the new truth.
        print("", file=sys.stderr)
        for line in sorted(mismatches):
            print(f"MISMATCH  {line}", file=sys.stderr)
        print(
            f"\n{len(mismatches)} file(s) differ from {receipt_path}. The receipt was "
            f"left untouched. Delete the offending files to re-fetch them, or say why "
            f"the corpus changed before you update the receipt.",
            file=sys.stderr,
        )
        return 1

    write_receipt(receipt_path, hashes)
    print(f"receipt: {len(hashes)} files in {receipt_path}")

    if dev_manifest is not None:
        micro = build_micro(dev_manifest, data_root)
        frames = sum(len(w["frames"]) for w in micro["windows"])
        print(
            f"micro:   {micro['n_windows']} windows, {frames} frames in "
            f"{data_root / 'micro'}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
