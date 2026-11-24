"""Download the COCO checkpoints for the three project models from HF Mirror."""

import shutil
from pathlib import Path
from urllib.request import urlopen


MIRROR = "https://hf-mirror.com"
WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
CHECKPOINTS = (
    ("Ultralytics/YOLOv5", "553ca28641b8c2129596b0ecc50fc0902f7bd7ac", "yolov5s.pt"),
    ("Ultralytics/YOLOv5", "553ca28641b8c2129596b0ecc50fc0902f7bd7ac", "yolov5n.pt"),
    ("kadirnar/yolov7-tiny-v0.1", "23120e9fcffb2beedbbaf5c992f0ae1bd1485a57", "yolov7-tiny.pt"),
)


def main():
    WEIGHTS.mkdir(exist_ok=True)
    for repo, revision, filename in CHECKPOINTS:
        target = WEIGHTS / filename
        if target.is_file():
            print(f"{filename}: already present")
            continue

        url = f"{MIRROR}/{repo}/resolve/{revision}/{filename}"
        partial = target.with_suffix(".part")
        try:
            with urlopen(url, timeout=30) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print(f"{filename}: downloaded")


if __name__ == "__main__":
    main()
