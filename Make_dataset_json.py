#!/usr/bin/env python3
"""
Create nnU-Net dataset.json for an existing nnUNet_raw dataset folder.

Example:
  python make_dataset_json.py \
    --root /path/to/nnUNet_raw/Dataset001_TotalSegmentatorPancreas \
    --name Dataset001_TotalSegmentatorPancreas \
    --labels background=0 pancreas=1 \
"""

import os
import json
import argparse
from typing import Dict, List, Tuple


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    """
    Parse a list like ["background=0", "pancreas=1"] into {"background":"0","pancreas":"1"}.
    Values are kept as strings to match common nnU-Net dataset.json conventions.
    """
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected KEY=VALUE format, got: {it}")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f"Invalid KEY=VALUE pair: {it}")
        out[k] = v
    return out


def list_files(folder: str, suffix: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted([f for f in os.listdir(folder) if f.endswith(suffix)])


def infer_dataset_name_from_root(root: str) -> str:
    return os.path.basename(os.path.normpath(root))


def validate_pairs(train_images: List[str], train_labels: List[str]) -> List[Tuple[str, str]]:
    """
    For each image 'XXXX_0000.nii.gz' ensure a label 'XXXX.nii.gz' exists.
    Returns list of (img, label) pairs.
    """
    label_set = set(train_labels)
    pairs: List[Tuple[str, str]] = []
    missing = []

    for img in train_images:
        case_id = img.split("_")[0]
        lbl = f"{case_id}.nii.gz"
        if lbl not in label_set:
            missing.append((img, lbl))
        else:
            pairs.append((img, lbl))

    if missing:
        preview = "\n".join([f"  image={i}  expected_label={l}" for i, l in missing[:10]])
        raise RuntimeError(
            f"Missing labels for {len(missing)} training images. First examples:\n{preview}"
        )

    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Create dataset.json inside an nnUNet_raw dataset folder."
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to nnUNet_raw dataset folder (contains imagesTr/imagesTs/labelsTr).",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name stored in dataset.json. Default: basename of --root",
    )
    p.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text dataset description.",
    )
    p.add_argument("--reference", type=str, default="")
    p.add_argument("--licence", type=str, default="")
    p.add_argument("--release", type=str, default="1.0")
    p.add_argument(
        "--tensorImageSize",
        type=str,
        default="3D",
        help='Usually "3D".',
    )
    p.add_argument(
        "--modality",
        type=str,
        default="CT",
        help='Modality string, e.g. "CT", "MR". Used for modality/channel_names.',
    )
    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help='Label mapping as KEY=VALUE pairs, e.g. background=0 pancreas=1',
    )
    p.add_argument(
        "--file_ending",
        type=str,
        default=".nii.gz",
        help='Usually ".nii.gz".',
    )
    p.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help='Output filename inside --root (default: "dataset.json").',
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Do everything except writing the JSON file.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    root = os.path.abspath(args.root)
    images_tr_dir = os.path.join(root, "imagesTr")
    images_ts_dir = os.path.join(root, "imagesTs")
    labels_tr_dir = os.path.join(root, "labelsTr")

    name = args.name or infer_dataset_name_from_root(root)
    labels = parse_kv_list(args.labels)

    # List files
    train_images = list_files(images_tr_dir, "_0000.nii.gz")
    test_images = list_files(images_ts_dir, "_0000.nii.gz") if os.path.isdir(images_ts_dir) else []
    train_labels = list_files(labels_tr_dir, ".nii.gz")

    # Validate that every training image has a corresponding label
    train_pairs = validate_pairs(train_images, train_labels)

    dataset = {
        "name": name,
        "description": args.description,
        "reference": args.reference,
        "licence": args.licence,
        "release": args.release,
        "tensorImageSize": args.tensorImageSize,
        "modality": {"0": args.modality},
        "channel_names": {"0": args.modality},
        "file_ending": args.file_ending,
        "labels": labels,
        "numTraining": len(train_pairs),
        "numTest": len(test_images),
        "training": [
            {"image": f"./imagesTr/{img}", "label": f"./labelsTr/{lbl}"}
            for img, lbl in train_pairs
        ],
        "test": [f"./imagesTs/{img}" for img in test_images],
    }

    out_path = os.path.join(root, args.output)
    if args.dry_run:
        print("Dry run: would write dataset.json to:")
        print(out_path)
        print(f"Found {len(train_pairs)} training cases and {len(test_images)} test cases.")
        return

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Created: {out_path}")
    print(f"Training cases: {len(train_pairs)}")
    print(f"Test cases: {len(test_images)}")


if __name__ == "__main__":
    main()
