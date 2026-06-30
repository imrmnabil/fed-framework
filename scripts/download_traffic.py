#!/usr/bin/env python
"""Download the GTSRB training images and lay them out as data/gtsrb/Train/<class>/.

Pulls the official 'GTSRB_Final_Training_Images.zip' (INI Benchmark) and reorganizes
the per-class .ppm folders (00000..00042) into Train/0..42/ that the loader expects.
~263 MB download; GPU recommended for full-scale training (use the CPU subsample knob).
"""
import _bootstrap  # noqa: F401

import shutil
import urllib.request
import zipfile
from pathlib import Path

from fedcity.config import REPO_ROOT

URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Training_Images.zip"
)


def main():
    data_dir = REPO_ROOT / "data" / "gtsrb"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "gtsrb_train.zip"
    train_dir = data_dir / "Train"

    if not zip_path.exists():
        print(f"Downloading GTSRB training images (~263 MB) from {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    # Source layout: GTSRB/Final_Training/Images/000NN/*.ppm
    images_root = next(data_dir.glob("**/Final_Training/Images"), None)
    if images_root is None:
        raise SystemExit("Could not locate Final_Training/Images after extraction.")

    train_dir.mkdir(exist_ok=True)
    moved = 0
    for class_folder in sorted(images_root.iterdir()):
        if not class_folder.is_dir():
            continue
        cls = int(class_folder.name)            # 00012 -> 12
        dst = train_dir / str(cls)
        dst.mkdir(exist_ok=True)
        for ppm in class_folder.glob("*.ppm"):
            shutil.copy(ppm, dst / ppm.name)
            moved += 1
    print(f"GTSRB ready: {moved} images across {len(list(train_dir.iterdir()))} classes -> {train_dir}")


if __name__ == "__main__":
    main()
