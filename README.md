# YOLOv5 TensorRT inference

在装有 CUDA、TensorRT Python 绑定、OpenCV 和 NumPy 的 Linux 设备上直接运行。输入模型为 batch 1 的静态 YOLOv5 检测 ONNX，引擎输出是 NMS 前的检测张量。输入分辨率取自引擎。

## 构建 FP32 引擎

```bash
bash scripts/benchmark_fp32.sh model.onnx model.engine
```

脚本从 PATH 查找 trtexec，可通过 `TRTEXEC` 指定程序路径。额外参数会传给 trtexec，JetPack 4.6 上可加 `--workspace=512`。已有本机可用的引擎时，可以直接运行推理。

## 推理并生成 COCO JSON

```bash
python3 scripts/detect.py model.engine images/ --output detections.json
```

图片参数接受单张文件或目录。数字文件名用作图像 ID，其他文件按顺序编号。传入 `--annotations` 可沿用 COCO 标注中的原始 ID。80 类模型默认采用 COCO 类别映射，其他模型可以按模型类别顺序传入 `--category-ids`。

输出使用 COCO 的原图像素坐标 `[x, y, width, height]`。用 `--conf` 和 `--iou` 调整默认的 0.25 置信度阈值及 0.45 NMS IoU 阈值。程序统计从已解码图片到检测结果的平均耗时和 FPS。系统找不到 CUDA 运行库时，可设置 `CUDART_LIBRARY` 指向实际库文件。
