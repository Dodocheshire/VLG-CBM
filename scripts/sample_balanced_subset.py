#!/usr/bin/env python3
"""Create deterministic, nested, class-balanced image subset manifests."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def stable_rank(seed: int, relative_path: str) -> bytes:
    value = f"{seed}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(value).digest()


def read_places365(source_root: Path, file_list: Path):
    samples = defaultdict(list)
    with file_list.open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                relative_path, target = line.strip().rsplit(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed Places365 row {line_number}: {line!r}") from exc
            samples[int(target)].append(relative_path.lstrip("/"))
    return samples


def read_imagenet(source_root: Path):
    samples = defaultdict(list)
    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for target, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                samples[target].append(path.relative_to(source_root).as_posix())
    return samples


def read_classes(path: Path):
    with path.open("r") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["places365", "imagenet"], required=True)
    parser.add_argument("--images-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--output-dir", default="datasets/subsets")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--classes-file", default=None)
    args = parser.parse_args()

    if args.images_per_class <= 0:
        parser.error("--images-per-class must be positive")

    if args.dataset == "places365":
        source_root = Path(args.source_root or "datasets/places365_torch/data_256")
        file_list = Path(
            args.file_list or "datasets/places365_torch/places365_train_standard.txt"
        )
        classes_file = Path(args.classes_file or "concept_files/places365_classes.txt")
        samples_by_class = read_places365(source_root, file_list)
    else:
        source_root = Path(
            args.source_root or "datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
        )
        classes_file = Path(args.classes_file or "concept_files/imagenet_classes.txt")
        if not source_root.is_dir():
            raise FileNotFoundError(f"ImageNet training root not found: {source_root}")
        samples_by_class = read_imagenet(source_root)

    classes = read_classes(classes_file)
    if len(samples_by_class) != len(classes):
        raise ValueError(
            f"Found {len(samples_by_class)} image classes but {len(classes)} class names"
        )

    selected_by_class = []
    for target in range(len(classes)):
        candidates = samples_by_class[target]
        if len(candidates) < args.images_per_class:
            raise ValueError(
                f"Class {target} ({classes[target]}) has only {len(candidates)} images"
            )
        ranked = sorted(candidates, key=lambda path: stable_rank(args.seed, path))
        selected_by_class.append(ranked[: args.images_per_class])

    # Rank-major ordering makes the N-images/class manifest an exact index prefix
    # of every larger manifest generated with the same seed.
    selected = [
        (selected_by_class[target][rank], target)
        for rank in range(args.images_per_class)
        for target in range(len(classes))
    ]

    missing = [path for path, _ in selected if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected source image is missing: {missing[0]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.output_name}.tsv"
    metadata_path = output_dir / f"{args.output_name}.json"

    with manifest_path.open("w") as handle:
        for relative_path, target in selected:
            handle.write(f"{relative_path}\t{target}\n")

    metadata = {
        "schema_version": 1,
        "dataset": args.dataset,
        "output_name": args.output_name,
        "source_root": os.path.relpath(source_root.resolve(), output_dir.resolve()),
        "classes_file": str(classes_file),
        "classes": classes,
        "seed": args.seed,
        "images_per_class": args.images_per_class,
        "class_count": len(classes),
        "sample_count": len(selected),
        "selection": "per-class SHA-256 rank; smaller subsets are prefixes of larger subsets",
    }
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "metadata": str(metadata_path),
                "classes": len(classes),
                "samples": len(selected),
            }
        )
    )


if __name__ == "__main__":
    main()
