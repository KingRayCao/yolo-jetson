"""Extract the official COCO2017 archives when needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1] / "COCO2017"
    for name, folder in (
        ("annotations_trainval2017.zip", "annotations"),
        ("val2017.zip", "val2017"),
        ("train2017.zip", "train2017"),
    ):
        if (root / folder).is_dir():
            print(f"{folder}: already present")
            continue
        print(f"Extracting {name}", flush=True)
        with ZipFile(root / name) as archive, TemporaryDirectory(dir=root) as staged:
            archive.extractall(staged)
            (Path(staged) / folder).rename(root / folder)


if __name__ == "__main__":
    main()
