"""Cut and concatenate kept segments into a single output video via ffmpeg."""

import shutil
import subprocess
from typing import List, Tuple

import imageio_ffmpeg

from .segments import Segment


def _ffmpeg_binary() -> str:
    # Prefer a system ffmpeg if the user has one on PATH; otherwise fall back
    # to the binary bundled by imageio-ffmpeg so no manual install is needed.
    system_ffmpeg = shutil.which("ffmpeg")
    return system_ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()


def build_filter_complex(segments: List[Segment], with_audio: bool) -> Tuple[str, List[str]]:
    """Build a -filter_complex graph that trims each segment and concats them
    in order. Returns (filter_str, map_args)."""
    parts = []
    concat_inputs = []
    for i, seg in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS[v{i}];"
        )
        if with_audio:
            parts.append(
                f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
        else:
            concat_inputs.append(f"[v{i}]")

    n = len(segments)
    a_flag = 1 if with_audio else 0
    parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a={a_flag}[outv]" + ("[outa]" if with_audio else ""))
    filter_complex = "".join(parts)

    map_args = ["-map", "[outv]"]
    if with_audio:
        map_args += ["-map", "[outa]"]
    return filter_complex, map_args


def _run_cut(ffmpeg: str, input_path: str, segments: List[Segment], output_path: str, with_audio: bool) -> None:
    filter_complex, map_args = build_filter_complex(segments, with_audio)
    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        *map_args,
        "-c:v", "libx264",
    ]
    if with_audio:
        cmd += ["-c:a", "aac"]
    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True)


def cut_and_concat(input_path: str, segments: List[Segment], output_path: str) -> None:
    if not segments:
        raise ValueError("No segments to cut - nothing was detected as live play.")

    ffmpeg = _ffmpeg_binary()
    try:
        _run_cut(ffmpeg, input_path, segments, output_path, with_audio=True)
    except subprocess.CalledProcessError:
        # Most likely the source has no audio stream - retry video-only.
        _run_cut(ffmpeg, input_path, segments, output_path, with_audio=False)
