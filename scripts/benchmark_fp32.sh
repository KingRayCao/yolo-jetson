#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    printf 'Usage: bash %s MODEL.onnx OUTPUT.engine [trtexec options...]\n' "$0" >&2
    exit 2
fi

model=$1
engine=$2
shift 2
trtexec="${TRTEXEC:-$(command -v trtexec || true)}"
trtexec="${trtexec:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p -- "$(dirname -- "$engine")"
"$trtexec" --onnx="$model" --saveEngine="$engine" --noTF32 \
    --iterations=200 --duration=0 --warmUp=500 --avgRuns=100 --useSpinWait "$@"
