#!/usr/bin/env bash
set -euo pipefail

# Convert all audio files in this directory to compressed MP3 mono.
# Defaults are optimized for IVR voice prompts (small file size + clear speech).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$SCRIPT_DIR/original_files}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/converted_mp3}"
BITRATE="${BITRATE:-64k}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob

for input in "$INPUT_DIR"/*; do
  [ -f "$input" ] || continue

  ext="${input##*.}"
  ext_lower="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  case "$ext_lower" in
    wav|mp3|m4a|aac|flac|ogg|wma|aiff|opus)
      base="$(basename "$input")"
      stem="${base%.*}"
      output="$OUTPUT_DIR/$stem.mp3"
      echo "Converting: $base -> $(basename "$output")"
      ffmpeg -y -i "$input" -ac 1 -ar "$SAMPLE_RATE" -b:a "$BITRATE" "$output"
      ;;
    *)
      ;;
  esac
done

echo "Done. Converted files are in: $OUTPUT_DIR"
