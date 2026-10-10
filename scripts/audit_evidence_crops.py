#!/usr/bin/env python3
"""Does the evidence crop show the fire? -- one lap of the live wall, every DEV fire's first hit.

    .venv/bin/python scripts/audit_evidence_crops.py [--base https://live.plumerank.edycu.dev]
    # writes results/evidence_first_hits_<date>.json and results/evidence_first_hits_<date>_{crops,full}.jpg

WHY
---
Every watch-list row ships an evidence crop -- the camera's largest tracked foreground
component, boxed, with the pinned-foot and climbing-top lines -- and the README calls it the
safeguard that lets a human overrule the ranking in one glance. Whether that crop actually
frames the plume was never measured. The ranking is per camera, so a fire camera can reach
the top while its largest moving component is a cloud.

WHAT
----
1. Listen to the live wall's SSE stream for one full lap (it is deterministic lap to lap).
2. For every DEV (fire) window, take the first tick at or after ignition where the camera is
   rank 1 or fires an alert and published an evidence crop.
3. Fetch that crop from the box, locate it in the working-resolution frame by template
   matching (``cv2.matchTemplate``, TM_CCOEFF_NORMED), and cut the same region from the
   full-resolution HPWREN JPEG, so the region can be judged at the camera's own resolution.
4. Write a JSON of hits and two contact sheets (the crops as served; the same regions at full
   resolution). The by-eye label of each region -- does it contain the plume? -- lives in
   ``results/evidence_first_hits_labels.json``, written by a person looking at the full-res
   sheet, and is merged in when present. Nothing here is a detector: it is an audit.

Read-only against the pipeline; never opens HOLD (the wall roster is DEV + CONFUSE).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request
from datetime import date

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import _align, _decode_working_frame, _wall_sequences  # noqa: E402

LAP_TIMEOUT_S = 200


def _sse_lap(base: str) -> tuple[dict, dict[int, dict[str, dict]]]:
    """(meta, {frame_index: {camera_id: entry}}) for one full lap of the live wall."""
    req = urllib.request.Request(base + "/api/watchlist", headers={"Accept": "text/event-stream"})
    meta: dict = {}
    frames: dict[int, dict[str, dict]] = {}
    event = None
    with urllib.request.urlopen(req, timeout=LAP_TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                doc = json.loads(line[5:])
                if event == "meta":
                    meta = doc
                elif "frame_index" in doc:
                    frames[doc["frame_index"]] = {e["camera_id"]: e for e in doc["entries"]}
                    if meta and len(frames) >= meta["frame_count"]:
                        return meta, frames
    raise SystemExit("the stream ended before a full lap was seen")


def _fetch(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=20) as r:
        buf = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"undecodable evidence crop at {url}")
    return img


def _sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", default="https://live.plumerank.edycu.dev")
    args = ap.parse_args()

    meta, lap = _sse_lap(args.base)
    t0 = int(meta["t0_index"])
    arm = {s["camera_id"]: s["arm"] for s in meta["sequences"]}
    sid = {s["camera_id"]: s["sequence_id"] for s in meta["sequences"]}
    sequences, _ = _wall_sequences()
    cameras, _, _ = _align(sequences)
    by_id = {c.camera_id: c for c in cameras}

    labels_path = ROOT / "results" / "evidence_first_hits_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else {}

    hits, crops, regions = [], [], []
    for cam_id in meta["cameras"]:
        if arm.get(cam_id) != "dev":
            continue
        for f in range(t0, int(meta["frame_count"])):
            e = lap[f][cam_id]
            if (e["rank"] == 1 or e["alerted"]) and e.get("evidence_url"):
                break
        else:
            continue
        ev = _fetch(args.base + e["evidence_url"].split("?")[0])
        cam = by_id[cam_id]
        path = cam.sequence.frame_paths[cam.frame_offset + f]
        work = _decode_working_frame(path)
        full = cv2.imread(path, cv2.IMREAD_COLOR)
        if full is None:
            raise SystemExit(f"unreadable corpus frame {path} -- run `make fetch`")
        # The crop is served upscaled (rank.EVIDENCE_RENDER_SCALE); match at working scale.
        k = max(1, round(ev.shape[1] / 160)) if ev.shape[1] > work.shape[1] // 2 else 1
        tmpl = cv2.resize(ev, (ev.shape[1] // k, ev.shape[0] // k), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(work, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x0, y0) = cv2.minMaxLoc(res)
        sx, sy = full.shape[1] / work.shape[1], full.shape[0] / work.shape[0]
        th, tw = tmpl.shape[:2]
        box = [int(x0 * sx), int(y0 * sy), int((x0 + tw) * sx), int((y0 + th) * sy)]
        caption = f"{cam_id} T+{f - t0} rank {e['rank']}"
        crops.append(_caption(cv2.resize(ev, (320, 240)), caption))
        regions.append(_caption(cv2.resize(full[box[1] : box[3], box[0] : box[2]], (480, 360)), caption))
        hits.append(
            {
                "camera_id": cam_id,
                "sequence_id": sid[cam_id],
                "frame_index": f,
                "minutes_after_ignition": f - t0,
                "rank": e["rank"],
                "alerted": e["alerted"],
                "s1_s2_s3": [e["s1"], e["s2"], e["s3"]],
                "match_score": round(float(score), 3),
                "region_full_res_xyxy": box,
                "label": labels.get(cam_id),
            }
        )

    today = date.today().isoformat()
    out = ROOT / "results" / f"evidence_first_hits_{today}"
    cv2.imwrite(f"{out}_crops.jpg", _sheet(crops, 6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(f"{out}_full.jpg", _sheet(regions, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    counts: dict[str, int] = {}
    for h in hits:
        counts[str(h["label"])] = counts.get(str(h["label"]), 0) + 1
    payload = {
        "what": "evidence crop at each DEV fire's first post-ignition rank-1 or alert tick, one live lap",
        "base": args.base,
        "replay_id": meta.get("replay_id"),
        "label_meaning": "does the full-resolution crop region contain the plume, judged by eye",
        "label_counts": counts,
        "hits": hits,
    }
    pathlib.Path(f"{out}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(pathlib.Path(f"{out}.json").relative_to(ROOT), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
