"""Run a YOLOv5 TensorRT engine on images and write COCO-format detections."""

import argparse
import ctypes as ct
import json
import os
import time
from contextlib import ExitStack
from ctypes.util import find_library
from pathlib import Path

import cv2
import numpy as np


COCO_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def prepare(image, shape, dtype):
    height, width = image.shape[:2]
    target_h, target_w = shape
    gain = min(target_h / height, target_w / width)
    resized = int(round(width * gain)), int(round(height * gain))
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    x, y = (target_w - resized[0]) / 2, (target_h - resized[1]) / 2
    padded = cv2.copyMakeBorder(
        image, round(y - 0.1), round(y + 0.1), round(x - 0.1), round(x + 0.1),
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    batch = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=dtype)
    batch /= 255
    return batch, gain, (x, y)


def detections(predictions, image_id, image_shape, gain, pad, confidence, iou, category_ids):
    rows = predictions[0][predictions[0, :, 4] > confidence]
    class_scores = rows[:, 4:5] * rows[:, 5:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(rows)), classes]
    confident = scores > confidence
    rows, classes, scores = rows[confident], classes[confident], scores[confident]

    xy, half_size = rows[:, :2], rows[:, 2:4] / 2
    boxes = np.concatenate((xy - half_size, xy + half_size), axis=1)
    areas = rows[:, 2] * rows[:, 3]
    pending = np.argsort(-scores)
    kept = []
    while pending.size and len(kept) < 300:
        best, others = pending[0], pending[1:]
        kept.append(best)
        overlap_wh = np.maximum(
            0, np.minimum(boxes[best, 2:], boxes[others, 2:])
            - np.maximum(boxes[best, :2], boxes[others, :2]),
        )
        intersection = overlap_wh[:, 0] * overlap_wh[:, 1]
        union = areas[best] + areas[others] - intersection
        overlap = intersection / np.maximum(union, 1e-7)
        pending = others[(classes[others] != classes[best]) | (overlap <= iou)]

    original = (boxes[kept] - np.array(pad * 2, dtype=boxes.dtype)) / gain
    height, width = image_shape[:2]
    original[:, 0::2] = original[:, 0::2].clip(0, width)
    original[:, 1::2] = original[:, 1::2].clip(0, height)
    original[:, 2:] -= original[:, :2]
    return [
        {"image_id": image_id, "category_id": category_ids[classes[index]],
         "bbox": box.tolist(), "score": float(scores[index])}
        for index, box in zip(kept, original)
    ]


def cuda_check(code):
    if code:
        raise RuntimeError("CUDA runtime error {}".format(code))


def run(engine_path, images, confidence, iou, category_ids):
    import tensorrt as trt

    cuda = ct.CDLL(os.environ.get("CUDART_LIBRARY") or find_library("cudart") or "libcudart.so")
    cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    cuda.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    cuda.cudaFree.argtypes = [ct.c_void_p]

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Could not load the TensorRT engine")
    context = engine.create_execution_context()
    binding_api = hasattr(engine, "num_bindings")
    if binding_api:
        tensors = list(range(engine.num_bindings))
        inputs = [tensor for tensor in tensors if engine.binding_is_input(tensor)]
        shape_of, dtype_of = engine.get_binding_shape, engine.get_binding_dtype
    else:
        tensors = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        inputs = [tensor for tensor in tensors if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT]
        shape_of, dtype_of = engine.get_tensor_shape, engine.get_tensor_dtype
    input_tensor, = inputs
    output_tensor, = [tensor for tensor in tensors if tensor != input_tensor]
    input_shape = tuple(shape_of(input_tensor))
    if input_shape[:2] != (1, 3) or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("Expected a static 1x3xHxW input")
    input_dtype = trt.nptype(dtype_of(input_tensor))
    output = np.empty(tuple(shape_of(output_tensor)), dtype=trt.nptype(dtype_of(output_tensor)))
    classes = output.shape[-1] - 5
    if category_ids is None:
        category_ids = COCO_IDS if classes == len(COCO_IDS) else range(1, classes + 1)
    if len(category_ids) != classes:
        raise ValueError("Provide one category ID per model class")

    results = []
    elapsed = 0.0
    with ExitStack() as buffers:
        addresses = {}
        input_bytes = int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize
        for tensor, size in ((input_tensor, input_bytes), (output_tensor, output.nbytes)):
            address = ct.c_void_p()
            cuda_check(cuda.cudaMalloc(ct.byref(address), size))
            buffers.callback(cuda.cudaFree, address)
            addresses[tensor] = address.value
        if binding_api:
            bindings = [addresses[tensor] for tensor in tensors]
        else:
            for tensor, address in addresses.items():
                context.set_tensor_address(tensor, address)

        def infer(batch):
            cuda_check(cuda.cudaMemcpy(addresses[input_tensor], batch.ctypes.data, batch.nbytes, HOST_TO_DEVICE))
            executed = context.execute_v2(bindings) if binding_api else context.execute_async_v3(0)
            if not executed:
                raise RuntimeError("TensorRT inference failed")
            cuda_check(cuda.cudaMemcpy(output.ctypes.data, addresses[output_tensor], output.nbytes, DEVICE_TO_HOST))

        for number, (path, image_id) in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Could not read {}".format(path))
            if number == 0:
                warmup, _, _ = prepare(image, input_shape[-2:], input_dtype)
                infer(warmup)
            start = time.perf_counter()
            batch, gain, pad = prepare(image, input_shape[-2:], input_dtype)
            infer(batch)
            results.extend(detections(output, image_id, image.shape, gain, pad, confidence, iou, category_ids))
            elapsed += time.perf_counter() - start
    return results, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("images", type=Path, help="Image or directory")
    parser.add_argument("--output", type=Path, default=Path("detections.json"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--annotations", type=Path, help="Use image IDs from COCO annotations")
    parser.add_argument("--category-ids", type=int, nargs="+", help="Category IDs in model class order")
    args = parser.parse_args()

    images = [args.images]
    if args.images.is_dir():
        images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        parser.error("No images found")
    if args.annotations:
        with args.annotations.open(encoding="utf-8") as file:
            ids = {image["file_name"]: image["id"] for image in json.load(file)["images"]}
        image_ids = [ids[path.name] for path in images]
    elif all(path.stem.isdecimal() for path in images):
        image_ids = [int(path.stem) for path in images]
    else:
        image_ids = range(1, len(images) + 1)
    results, elapsed = run(args.engine, list(zip(images, image_ids)), args.conf, args.iou, args.category_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file)
    print("{} images, {} detections".format(len(images), len(results)))
    print("End-to-end: {:.2f} ms/image, {:.2f} FPS".format(1000 * elapsed / len(images), len(images) / elapsed))
    print("COCO JSON: {}".format(args.output))


if __name__ == "__main__":
    main()
