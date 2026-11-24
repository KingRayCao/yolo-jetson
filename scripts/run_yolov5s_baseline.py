"""Export YOLOv5s and verify FP32 detection parity on one COCO image.

Prepare the upstream checkout with:
    git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git runs/yolov5s_baseline/yolov5-v7.0
Run this script with the project's Python environment.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "yolov5s_baseline"
SOURCE = RUN_DIR / "yolov5-v7.0"
WEIGHTS = ROOT / "weights" / "yolov5s.pt"
IMAGE = ROOT / "COCO2017" / "val2017" / "000000000139.jpg"
INPUT_SIZE = (640, 640)
OPSET = 12
CONFIDENCE = 0.25
NMS_IOU = 0.45


def main() -> None:
    if not (SOURCE / ".git").exists():
        raise SystemExit(f"Prepare the YOLOv5 v7.0 checkout at {SOURCE}. See the script docstring.")
    os.environ["YOLOV5_CONFIG_DIR"] = str(RUN_DIR / "config")
    sys.path.insert(0, str(SOURCE))
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    # The v7.0 checkpoint contains a serialized PyTorch model.
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval().fuse()
    model.model[-1].export = True

    image = cv2.imread(str(IMAGE))
    if image is None:
        raise SystemExit(f"Cannot read COCO validation image {IMAGE}")
    padded, gain, pad = letterbox(image, INPUT_SIZE, auto=False)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    batch = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    tensor = torch.from_numpy(batch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        pytorch_output = model.to(device)(tensor.to(device))[0].cpu().numpy()
    model.cpu()

    onnx_name = "yolov5s_fp32.onnx"
    report_name = "baseline.json"
    with TemporaryDirectory(dir=RUN_DIR) as temporary:
        onnx_path = Path(temporary) / onnx_name
        torch.onnx.export(
            model, tensor, str(onnx_path), opset_version=OPSET,
            input_names=["images"], output_names=["output0"], dynamo=False,
        )
        # Fold the fixed Resize sizes before TensorRT 8.2 parses the graph.
        simplified, equivalent = simplify(onnx.load(onnx_path))
        if not equivalent:
            raise RuntimeError("ONNX simplification verification failed")
        onnx.checker.check_model(simplified)
        onnx.save(simplified, onnx_path)
        session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != list(batch.shape):
            raise RuntimeError("ONNX input shape differs from the fixed model input")
        onnx_output, = session.run(["output0"], {"images": batch})
        if pytorch_output.shape != (1, 25200, 85) or onnx_output.shape != pytorch_output.shape:
            raise RuntimeError("Unexpected YOLOv5s output shape")

        detections = []
        for output in (pytorch_output, onnx_output):
            result, = non_max_suppression(torch.from_numpy(output), CONFIDENCE, NMS_IOU, max_det=300)
            scale_boxes(INPUT_SIZE, result[:, :4], image.shape, ratio_pad=(gain, pad))
            detections.append(result.numpy())

        pytorch_boxes, onnx_boxes = detections
        if not len(pytorch_boxes) or pytorch_boxes.shape != onnx_boxes.shape:
            raise RuntimeError("Expected matching nonempty detections from both backends")
        np.testing.assert_array_equal(pytorch_boxes[:, 5], onnx_boxes[:, 5])
        np.testing.assert_allclose(pytorch_boxes[:, :4], onnx_boxes[:, :4], rtol=0, atol=0.05)
        np.testing.assert_allclose(pytorch_boxes[:, 4], onnx_boxes[:, 4], rtol=0, atol=1e-4)
        difference = np.abs(pytorch_boxes - onnx_boxes)
        report = {
            "image": IMAGE.relative_to(ROOT).as_posix(),
            "yolov5_version": "v7.0",
            "pytorch_version": torch.__version__,
            "pytorch_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "onnxruntime_version": ort.__version__,
            "onnxruntime_provider": session.get_providers()[0],
            "onnx_opset": OPSET,
            "onnx_simplified": True,
            "input_shape": list(batch.shape),
            "output_shape": list(pytorch_output.shape),
            "confidence_threshold": CONFIDENCE,
            "nms_iou_threshold": NMS_IOU,
            "detection_count": len(pytorch_boxes),
            "max_box_difference_px": float(difference[:, :4].max()),
            "max_confidence_difference": float(difference[:, 4].max()),
            "detections": [
                {
                    "label": model.names[int(pytorch[5])],
                    "pytorch_xyxy_confidence": pytorch[:5].tolist(),
                    "onnx_xyxy_confidence": exported[:5].tolist(),
                }
                for pytorch, exported in zip(pytorch_boxes, onnx_boxes)
            ],
        }
        report_path = Path(temporary) / report_name
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        onnx_path.replace(RUN_DIR / onnx_name)
        report_path.replace(RUN_DIR / report_name)

    print(f"FP32 parity passed on {IMAGE.name}: {report['detection_count']} detections")
    print(f"Maximum box difference: {report['max_box_difference_px']:.6g} px")
    print(f"Maximum confidence difference: {report['max_confidence_difference']:.6g}")
    print(f"Artifacts: {RUN_DIR}")


if __name__ == "__main__":
    main()
