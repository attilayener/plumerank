# data/micro/ -- provenance

Imagery: HPWREN, http://hpwren.ucsd.edu/

FIgLib carries no SPDX licence and requires a credit reference. It is used for EVALUATION ONLY and never for training -- test_no_figlib_in_training enforces that, and it is what makes every Californian number cross-network out-of-sample by construction.

## What these files are

10 replay windows (5 fire, 5 non-fire) of 31 frames each -- 310 JPEGs in total -- cut from 5 HPWREN FIgLib sequences on 5 distinct cameras.

Each frame was downscaled from 3072x2048 to 480x270 (6.4x in x, 7.6x in y -- the pipeline's working raster, which is 16:9 where the source is 3:2) with cv2.INTER_AREA, and re-encoded as JPEG at quality 98 by scripts/fetch.py. INTER_AREA and not the default bilinear: at this reduction bilinear samples four source pixels out of forty and aliases exactly the thin high-contrast structure the boundary statistic reads.

They exist so that `make verify` recomputes real numbers on real pixels in three minutes with no network and no credentials.

## Source sequences

- 20240622_Border58Fire_om-w-mobo-c
- 20240625_DelMarFire_tdlln-n-mobo-c
- 20240701_McCainfire_ws-e-mobo-c
- 20240724_GroveFire_tp-w-mobo-c
- 20240905_ValleyFire_cp-s-mobo-c

Each one was cut in two at its ignition frame, giving a `__pre` window and a `__fire` window. Every window's `source_sequence_id` in manifest_micro.json names the sequence above that it came from.

## Why the calibration windows are the pre-ignition halves, and what that means

`scripts/bench.py --arm micro` solves an alert threshold on the non-fire windows and measures time-to-detection on the fire ones. A corpus of fires only has no false-alarm denominator, so no threshold can be solved on it at all.

A FIgLib window spans -2400 s to +2400 s around the reported ignition, so its pre-ignition half is a real non-fire camera-night -- same camera, same time of day, same light. **micro's calibration windows are therefore the pre-ignition halves of its own fire sequences.** The fire half keeps 8 pre-ignition frames of lead-in so the background model is warm when the fire starts, and the non-fire half gives those frames up, so the two halves share no frame -- nothing the threshold is solved on is also part of a time-to-detection measurement. What that does not say: the micro *arm* is all 10 windows, the calibration ones included, so micro's arm false-alarm rate is measured partly on the frames its threshold was solved on. The time-to-detection numbers are clean; that rate is not, and it is not the published one.

One more thing this arm is not. Its time-to-detection is not an estimate: 5 fire windows is too few for a median, and which of them detect at all is chaotic with respect to noise well below one grey level. Re-encoding these same frames at JPEG quality 92/95/98/100 -- differences invisible on screen -- gives 0, 2, 2 and 1 detections of 5 and moves the solved threshold from 0.30 to 0.41, with identical code and an identical cut. What `make verify` reproduces is the committed bytes, byte for byte. It is a determinism check, not a measurement.

**This is why the micro arm proves that the pipeline runs and that its committed numbers are self-consistent, and why it is NOT -- and never claims to be -- the held-out headline.** The headline is measured on the 54 held-out sequences that survive exclusion -- data/manifest_hold.json names 60 windows, data/excluded.json drops 6 of them. Producing data/excluded.json meant a human looking at the FIRST FRAME of every HOLD window, about 40 minutes before ignition, against criteria stated before the pass; results/hold_access.log still records no HOLD read because no HOLD pixel has been scored., against the 33-window CONFUSE adversary set, by `make fetch && make bench` (about 95 minutes). `make verify` prints that number in a block labelled PUBLISHED, NOT RECOMPUTED HERE, quoted verbatim from results/.

## Licence

FIgLib carries no SPDX licence. Its terms ask for a credit reference, which is the line at the top of this file and appears again in the README, in the UI footer and in the SSE stream's meta event. This imagery is used for EVALUATION ONLY and never for training -- the one trained model in this project sees Pyro-SDIS (Apache-2.0) and nothing else, and `test_no_figlib_in_training` is what enforces it.

## Regenerating them

    python scripts/fetch.py --corpus figlib

The selection rule is in data/micro/manifest_micro.json under `selection`, and the cut is `micro_windows` in scripts/fetch.py.
