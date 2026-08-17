#!/usr/bin/env python3
"""Validate an H3 MP4 and render a six-frame visual contact sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--contact-sheet", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    args = parser.parse_args()

    with av.open(str(args.input)) as container:
        format_name = container.format.name
        video_stream = container.streams.video[0]
        video_codec = video_stream.codec_context.name
        video_width = int(video_stream.codec_context.width)
        video_height = int(video_stream.codec_context.height)
        stream_average_rate = str(video_stream.average_rate)
        video_frames = []
        video_times = []
        for index, frame in enumerate(container.decode(video=0)):
            video_frames.append((index, frame.to_image().convert("RGB")))
            video_times.append(float(frame.time) if frame.time is not None else None)

    with av.open(str(args.input)) as container:
        audio_stream = container.streams.audio[0]
        audio_codec = audio_stream.codec_context.name
        channels = int(audio_stream.codec_context.channels)
        sample_rate = int(audio_stream.codec_context.sample_rate)
        audio_arrays = []
        for frame in container.decode(audio=0):
            array = frame.to_ndarray()
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
            else:
                array = array.astype(np.float32)
            audio_arrays.append(array.reshape(-1))
    audio = np.concatenate(audio_arrays) if audio_arrays else np.empty(0, dtype=np.float32)

    if args.expected_frames is not None and len(video_frames) != args.expected_frames:
        raise ValueError(f"expected {args.expected_frames} decoded video frames, got {len(video_frames)}")
    if args.expected_width is not None and video_width != args.expected_width:
        raise ValueError(f"expected video width {args.expected_width}, got {video_width}")
    if args.expected_height is not None and video_height != args.expected_height:
        raise ValueError(f"expected video height {args.expected_height}, got {video_height}")
    if not video_frames:
        raise ValueError("video stream is empty")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("audio stream is empty or contains non-finite samples")

    selected_indices = [round(index * (len(video_frames) - 1) / 5) for index in range(6)]
    tiles = []
    frame_stats = []
    tile_width = 432
    tile_height = max(1, round(tile_width * video_height / video_width))
    for index in selected_indices:
        image = video_frames[index][1]
        array = np.asarray(image, dtype=np.float32) / 255.0
        frame_stats.append(
            {
                "index": index,
                "mean": float(array.mean()),
                "std": float(array.std()),
                "black_fraction": float((array.max(axis=2) < 0.02).mean()),
            }
        )
        tile = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 90, 24), fill=(0, 0, 0))
        draw.text((8, 5), f"frame {index}", fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (tile_width * 3, tile_height * 2), color=(16, 16, 16))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % 3) * tile_width, (position // 3) * tile_height))
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact_sheet, quality=92)

    finite_times = [value for value in video_times if value is not None]
    if len(finite_times) < 2:
        raise ValueError("video frames do not carry enough timestamps to validate duration")
    frame_intervals = np.diff(np.asarray(finite_times, dtype=np.float64))
    frame_interval = float(np.median(frame_intervals))
    derived_rate = 1.0 / frame_interval
    video_duration = finite_times[-1] - finite_times[0] + frame_interval
    audio_duration = audio.size / channels / sample_rate
    report = {
        "status": "pass",
        "input": str(args.input),
        "container": {
            "format": format_name,
            "size_bytes": args.input.stat().st_size,
        },
        "video": {
            "codec": video_codec,
            "width": video_width,
            "height": video_height,
            "frames": len(video_frames),
            "stream_average_rate": stream_average_rate,
            "derived_rate": derived_rate,
            "first_timestamp": finite_times[0],
            "last_timestamp": finite_times[-1],
            "duration_seconds": video_duration,
            "sampled_frame_stats": frame_stats,
        },
        "audio": {
            "codec": audio_codec,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": audio_duration,
            "decoded_scalar_samples": int(audio.size),
            "min": float(audio.min()),
            "max": float(audio.max()),
            "rms": float(math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))),
        },
        "contact_sheet": str(args.contact_sheet),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
