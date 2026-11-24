"""Validate the Jetson sample against ONNX Runtime and emit COCO detections."""


import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
import yaml

root = Path(__file__).resolve().parents[1]
run = root / "runs" / "yolov5s_baseline"
source = run / "yolov5-v7.0"
result = run / "jetson_fp32"
os.environ["YOLOV5_CONFIG_DIR"] = str(run / "config")
sys.path.insert(0, str(source))
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

with (run / "baseline.json").open(encoding="utf-8") as file:
    baseline = json.load(file)
with (source / "data/coco.yaml").open(encoding="utf-8") as file:
    names = yaml.safe_load(file)["names"]
image_path = root / baseline["image"]
image_id = int(image_path.stem)
image = cv2.imread(str(image_path))
_, ratio, pad = letterbox(image, (640, 640), auto=False)

tensor = np.fromfile(result / "sample_predictions.f32", dtype="<f4").reshape(baseline["output_shape"])
boxes, = non_max_suppression(
    torch.from_numpy(tensor), baseline["confidence_threshold"], baseline["nms_iou_threshold"], max_det=300,
)
scale_boxes((640, 640), boxes[:, :4], image.shape, ratio_pad=(ratio, pad))
boxes = boxes.numpy()
reference = baseline["detections"]
expected = np.array([row["onnx_xyxy_confidence"] for row in reference])
actual_labels = [names[int(row[5])] for row in boxes]
if actual_labels != [row["label"] for row in reference]:
    raise RuntimeError(f"TensorRT classes differ from ONNX Runtime: {actual_labels}")
np.testing.assert_allclose(boxes[:, :4], expected[:, :4], rtol=0, atol=0.05)
np.testing.assert_allclose(boxes[:, 4], expected[:, 4], rtol=0, atol=1e-4)
difference = np.abs(boxes[:, :5] - expected)

coco = COCO(str(root / "COCO2017" / "annotations" / "instances_val2017.json"))
if coco.imgs[image_id]["file_name"] != image_path.name:
    raise RuntimeError("Invalid COCO image ID")
category_ids = {category["name"]: category_id for category_id, category in coco.cats.items()}
detections = []
for (left, top, right, bottom, score, _), label in zip(boxes.tolist(), actual_labels):
    detections.append({
        "image_id": image_id,
        "category_id": category_ids[label],
        "bbox": [left, top, right - left, bottom - top],
        "score": score,
    })

temporary = result / "sample_coco.json.part"
try:
    temporary.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
    checked = coco.loadRes(str(temporary))
    if len(checked.getAnnIds(imgIds=[image_id])) != len(detections):
        raise RuntimeError("COCO API failed to reload the sample detections")
    temporary.replace(result / "sample_coco.json")
finally:
    temporary.unlink(missing_ok=True)
print(f"COCO-compatible TensorRT detections: {len(detections)}")
print(f"Maximum TensorRT vs ONNX Runtime box difference: {difference[:, :4].max():.6g} px")
print(f"Maximum TensorRT vs ONNX Runtime confidence difference: {difference[:, 4].max():.6g}")
