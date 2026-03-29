#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_video.mp4> <output_dir> [segment_sec]"
  exit 1
fi

INPUT="$1"
OUT_DIR="$2"
SEG_DUR="${3:-2}"

mkdir -p "$OUT_DIR"

# Multi-bitrate DASH ladder (720p / 480p / 360p / 240p)
ffmpeg -y -i "$INPUT" \
  -map 0:v:0 -map 0:v:0 -map 0:v:0 -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset veryfast -profile:v main -g $((SEG_DUR * 25)) -keyint_min $((SEG_DUR * 25)) -sc_threshold 0 \
  -b:v:0 2500k -maxrate:v:0 2675k -bufsize:v:0 3750k -s:v:0 1280x720 \
  -b:v:1 1200k -maxrate:v:1 1280k -bufsize:v:1 1800k -s:v:1 854x480 \
  -b:v:2 700k  -maxrate:v:2 750k  -bufsize:v:2 1100k -s:v:2 640x360 \
  -b:v:3 350k  -maxrate:v:3 400k  -bufsize:v:3 700k  -s:v:3 426x240 \
  -c:a aac -b:a 128k \
  -use_timeline 1 -use_template 1 -window_size 5 -adaptation_sets "id=0,streams=v id=1,streams=a" \
  -f dash -seg_duration "$SEG_DUR" "$OUT_DIR/manifest.mpd"

echo "DASH assets generated at: $OUT_DIR"
