# SportChopper

Automatically cut the dead time between points out of match footage (volleyball, tennis, pickleball), leaving a single trimmed video of just live play.

## How it works

1. **Detect.** A pretrained YOLOv8n model (COCO weights, downloaded automatically on first run — no training required) finds people and the sports ball in frames sampled from the video at a fixed analysis rate (default ~5 fps, independent of the source frame rate).
2. **Track.** A lightweight nearest-centroid tracker links detections across consecutive sampled frames to estimate ball speed and aggregate player movement.
3. **Score.** Ball presence/speed and player motion are combined into a single 0–1 "live play" activity score per sampled frame.
4. **Segment.** The score is smoothed and hysteresis-thresholded into play/dead segments, short blips are dropped, nearby segments are bridged across small gaps, and each kept segment is padded slightly so serves/rally starts aren't clipped.
5. **Cut.** The kept segments are trimmed and concatenated (video + audio) into a single output file with ffmpeg.

This is a heuristic v1: thresholds are tuned per sport in [`config.py`](src/sportchopper/config.py) but will likely need adjustment against real footage. Use `--dry-run` and `--debug-plot` (see below) to inspect what the detector is seeing before trusting a full cut.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

No separate ffmpeg install is required — `imageio-ffmpeg` bundles a binary that's used automatically if `ffmpeg` isn't already on your PATH.

## Usage

```powershell
sportchopper chop path\to\match.mp4 -o highlights.mp4 --sport volleyball
```

If you omit the input path, a native file picker opens so you can browse to a video instead:

```powershell
sportchopper chop --sport volleyball
```

> **Note on phone-connected videos:** if you pick a file from a phone shown under "This PC" (an MTP device, not a real drive letter), Windows' file dialog can only materialize a small preview stub for large files rather than the full video, so processing will fail on a truncated file. Copy the video to local disk first (drag it out in File Explorer, or use your phone's photo-sync app) and pick the local copy instead.

Useful flags:

- `--sport {volleyball,tennis,pickleball}` — selects the tuned preset (volleyball is the only one validated so far).
- `--dry-run` — analyze and print the detected point segments (start/end timestamps) as JSON, without producing an output video. Use this first on new footage.
- `--debug-plot out.png` — save a plot of the activity score over time with the enter/exit thresholds and kept segments overlaid, to visually sanity-check detection quality.
- `--conf`, `--sample-fps`, `--min-point-len`, `--min-gap`, `--pad` — override the sport preset's detector confidence, analysis frame rate, minimum point duration, minimum gap to bridge, and padding, respectively.

Example tuning workflow on a new clip:

```powershell
sportchopper chop match.mp4 --dry-run --debug-plot debug_activity.png
```

Inspect `debug_activity.png`; if points are being missed or merged incorrectly, adjust `--conf` or the thresholds in `config.py` and re-run.

## Project layout

```
src/sportchopper/
├── cli.py         # argparse entry point (the `sportchopper` command)
├── detection.py   # YOLOv8 wrapper: per-frame person/ball detections
├── tracking.py    # frame-to-frame centroid matching -> ball speed & player motion
├── activity.py    # turns tracked motion into a single 0-1 activity score
├── segments.py    # smoothing, hysteresis thresholding, merge/pad -> kept segments
├── cutter.py      # ffmpeg trim + concat into the final output video
├── video.py       # OpenCV frame sampling + video metadata
└── config.py      # per-sport tunable presets
```

`segments.py` and `activity.py` are pure logic with no video/model dependency, so they're covered by fast unit tests in `tests/` that run against synthetic data.

## Testing

```powershell
pip install -e ".[test]"  # or: pip install -r requirements.txt
pytest
```

The unit tests cover the scoring and segmentation logic only. There's no bundled sample video — validate end-to-end against your own footage with `--dry-run`/`--debug-plot` before trusting a full cut.

## Known limitations / roadmap

- Detection uses a generic pretrained "sports ball" class rather than a ball model trained specifically for volleyball/tennis/pickleball, so accuracy on fast, small, or motion-blurred balls will vary — expect to tune thresholds per camera setup.
- No whistle/audio cues are used yet; volleyball point boundaries in particular are often signaled by a whistle, which could improve accuracy in a future version.
- Tennis and pickleball presets in `config.py` are untuned starting points — volleyball is the only sport this has been designed and reasoned through end-to-end so far.
- Cutting always re-encodes (for frame-accurate trims) rather than stream-copying on keyframes, so exporting is slower than a simple `-c copy` cut.
