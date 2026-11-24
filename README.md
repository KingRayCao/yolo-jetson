# YOLOv5s on Jetson Nano

本项目在 Windows 导出 YOLOv5s v7.0 FP32 ONNX，在 Jetson Nano 上构建 TensorRT 8.2.1 引擎并验证单图结果。部署输入固定为 1×3×640×640，ONNX opset 为 12。

## Windows 准备与复现

需要 PowerShell 7.4 或更新版本、Git、OpenSSH、tar 和 uv。将 COCO2017 的三个原始 ZIP 放到项目的 COCO2017 目录，也可以使用该目录中已经解压的数据。Git 与 Python 下载使用当前进程配置的网络代理。

```powershell
pwsh -File scripts/prepare_local.ps1
```

脚本按 uv.lock 同步环境，下载缺少的权重，准备 COCO2017 并拉取 YOLOv5 v7.0 源码，再导出和简化 ONNX。runs/yolov5s_baseline/baseline.json 记录本机 PyTorch 和 ONNX Runtime 的结果。

## 同步与板端基线

准备可用的 SSH 主机别名 jetson。板端项目目录为 /home/nvidia/yolo-jetson。

```powershell
git push origin main
pwsh -File scripts/jetson.ps1 -Action sync
pwsh -File scripts/jetson.ps1 -Action check
pwsh -File scripts/jetson.ps1 -Action benchmark
pwsh -File scripts/jetson.ps1 -Action collect
```

- sync 用 Git bundle 将已提交的 main 同步到 Jetson，板端保持与 origin/main 的跟踪关系。它还会传输模型、参考结果和缺少的 COCO val2017 数据。仅同步代码时加 -CodeOnly。
- benchmark 在 Jetson 本地构建 FP32 引擎，用 COCO 图像 000000000139.jpg 的输入张量预热 500 ms，并测量 200 次推理。给 benchmark 与 collect 都加 -ReuseEngine 可复测已构建的同一模型。
- collect 拉取 trtexec 日志，在 Windows 用同一套 YOLOv5 后处理核对输出，并生成经 COCO API 验证的样例 sample_coco.json。日志和样例位于 runs/yolov5s_baseline/jetson_fp32。

trtexec 的计时覆盖引擎计算与张量传输。课程所需的端到端 FPS 和全量检测精度由完整测试集评估获得。Git 管理脚本、环境定义和锁文件。数据集、权重、上游源码副本和基线产物保存在本地或板端工作目录。
