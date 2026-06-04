#!/usr/bin/env bash
# Build the ownscribe-transcribe Swift binary (FluidAudio Parakeet ASR + diarization)
# and drop it into the repo-level bin/ directory next to ownscribe-audio.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_BIN="$(cd "$SCRIPT_DIR/../.." && pwd)/bin"

mkdir -p "$REPO_BIN"

echo "Building ownscribe-transcribe (release)..."
cd "$SCRIPT_DIR"
swift build -c release

PRODUCT="$SCRIPT_DIR/.build/release/ownscribe-transcribe"
if [[ ! -f "$PRODUCT" ]]; then
    echo "Build failed: $PRODUCT not found" >&2
    exit 1
fi

cp "$PRODUCT" "$REPO_BIN/ownscribe-transcribe"
echo "Built: $REPO_BIN/ownscribe-transcribe"
