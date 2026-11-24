#!/usr/bin/env bash
set -euo pipefail

case "$*" in
    '') mode=build ;;
    --reuse-engine) mode=reuse ;;
    *) exit 2 ;;
esac

cd "$(dirname "$0")/.."
run="$PWD/runs/yolov5s_baseline"
result="$run/jetson_fp32"
image="$PWD/COCO2017/val2017/000000000139.jpg"
engine="$result/yolov5s_fp32.engine"
trtexec=/usr/src/tensorrt/bin/trtexec
mkdir -p "$result"
scratch=$(mktemp -d "$result/.run.XXXXXX")
trap 'rm -r -- "$scratch"' EXIT

python3 -B - "$image" "$scratch/input.f32" <<'PYTHON'
import sys
import cv2
import numpy as np

image = cv2.imread(sys.argv[1])
if image is None:
    raise SystemExit("Could not read the reference JPEG")
height, width = image.shape[:2]
gain = min(640.0 / height, 640.0 / width)
new_width, new_height = int(round(width * gain)), int(round(height * gain))
if (new_width, new_height) != (width, height):
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
pad_x, pad_y = (640 - new_width) / 2.0, (640 - new_height) / 2.0
image = cv2.copyMakeBorder(
    image, int(round(pad_y - 0.1)), int(round(pad_y + 0.1)),
    int(round(pad_x - 0.1)), int(round(pad_x + 0.1)),
    cv2.BORDER_CONSTANT, value=(114, 114, 114),
)
batch = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
batch /= np.float32(255)
batch.tofile(sys.argv[2])
print("Input JPEG: {}x{}; benchmark batch: {}".format(width, height, batch.shape))
PYTHON

if [ "$mode" = build ]; then
    model_args=(--onnx="$run/yolov5s_fp32.onnx" --saveEngine="$scratch/yolov5s_fp32.engine" --workspace=512 --noTF32)
    log="$result/trtexec.log"
else
    model_args=(--loadEngine="$engine")
    log="$result/trtexec_reuse.log"
fi

"$trtexec" "${model_args[@]}" \
    --loadInputs="images:$scratch/input.f32" \
    --iterations=200 --duration=0 --warmUp=500 --avgRuns=100 --useSpinWait \
    --exportOutput="$scratch/output.json" 2>&1 | tee "$log"

python3 -B - "$scratch/output.json" "$scratch/sample_predictions.f32" <<'PYTHON'
import json
import sys
import numpy as np

with open(sys.argv[1]) as file:
    output = next(item for item in json.load(file) if item["name"] == "output0")
predictions = np.asarray(output["values"], dtype="<f4").reshape(1, 25200, 85)
predictions.tofile(sys.argv[2])
print("TensorRT output: 1x25200x85 FP32")
PYTHON

if [ "$mode" = build ]; then
    mv -- "$scratch/yolov5s_fp32.engine" "$engine"
fi
mv -- "$scratch/sample_predictions.f32" "$result/sample_predictions.f32"
printf 'FP32 engine: %s\nBenchmark log: %s\n' "$engine" "$log"
