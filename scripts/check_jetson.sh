#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '[System and NVIDIA packages]'
hostname
uname -mr
grep '^PRETTY_NAME=' /etc/os-release
head -n 1 /etc/nv_tegra_release
dpkg-query -W -f='${binary:Package} ${Version}\n' \
    nvidia-l4t-core cuda-toolkit-10-2 libcudnn8 libnvinfer8 libnvonnxparsers8 libopencv-dev

printf '\n%s\n' '[Compilers and runtime]'
gcc --version | sed -n '1p'
g++ --version | sed -n '1p'
cmake --version | sed -n '1p'
/usr/local/cuda/bin/nvcc --version
pkg-config --modversion opencv4
/usr/src/tensorrt/bin/trtexec --help | sed -n '1p'
python3 -B - <<'PYTHON'
import ctypes
import cv2
import numpy

print("OpenCV {}, NumPy {}".format(cv2.__version__, numpy.__version__))
cuda = ctypes.CDLL("libcuda.so.1")
count = ctypes.c_int()
if cuda.cuInit(0) or cuda.cuDeviceGetCount(ctypes.byref(count)) or not count.value:
    raise SystemExit("CUDA device is unavailable")
print("CUDA devices: {}".format(count.value))
PYTHON

printf '\n%s\n' '[Memory, swap and storage]'
free -h
swapon --show
df -h "$HOME"
