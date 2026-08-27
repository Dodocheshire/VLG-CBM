#!/usr/bin/env python3
"""Measure concept-support gains between baseline and expanded annotations."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def format_concept(value):
    value = value.lower()
    for token in "-,.()":
        value = value.replace(token, " ")
    if value.startswith("a "):
        value = value[2:]
    elif value.startswith("an "):
        value = value[3:]
    return " ".join(value.split())


def load_concepts(path: Path):
    with path.open("r") as handle:
        return list(dict.fromkeys(format_concept(line) for line in handle if line.strip()))


def count_image_support(annotation_dir: Path, max_index=None):
    counts = Counter()
    files = sorted(annotation_dir.glob("*.json"), key=lambda path: int(path.stem))
    if max_index is not None:
        files = [path for path in files if int(path.stem) < max_index]
    for path in files:
        with path.open("r") as handle:
            rows = json.load(handle)
        labels = {
            format_concept(row["label"])
            for row in rows[1:]
            if "label" in row
        }
        counts.update(labels)
    return counts, len(files)


def coverage_summary(counts, concepts):
    values = [counts[concept] for concept in concepts]
    return {
        "detected_at_least_1": sum(value >= 1 for value in values),
        "stable_at_least_5": sum(value >= 5 for value in values),
        "stable_at_least_10": sum(value >= 10 for value in values),
        "singletons": sum(value == 1 for value in values),
        "zero_support": sum(value == 0 for value in values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-file", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--expanded-max-index", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    concepts = load_concepts(Path(args.concept_file))
    baseline, baseline_images = count_image_support(Path(args.baseline_dir))
    expanded, expanded_images = count_image_support(
        Path(args.expanded_dir), args.expanded_max_index
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_zero = [concept for concept in concepts if baseline[concept] == 0]
    baseline_tail = [concept for concept in concepts if 1 <= baseline[concept] <= 5]
    metrics = {
        "candidate_concepts": len(concepts),
        "baseline_images": baseline_images,
        "expanded_images": expanded_images,
        "baseline": coverage_summary(baseline, concepts),
        "expanded": coverage_summary(expanded, concepts),
        "tail_concept_recall": {
            "baseline_zero_recovered": sum(expanded[c] > 0 for c in baseline_zero),
            "baseline_zero_total": len(baseline_zero),
            "baseline_low_frequency_1_to_5_reaching_10": sum(
                expanded[c] >= 10 for c in baseline_tail
            ),
            "baseline_low_frequency_1_to_5_total": len(baseline_tail),
        },
        "noise_robustness_support_proxy": {
            "note": "Support counts are a label-noise proxy, not a direct false-positive estimate.",
            "singleton_fraction_baseline": sum(baseline[c] == 1 for c in concepts)
            / len(concepts),
            "singleton_fraction_expanded": sum(expanded[c] == 1 for c in concepts)
            / len(concepts),
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    with (output_dir / "concept_support.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept", "baseline_image_support", "expanded_image_support"])
        for concept in concepts:
            writer.writerow([concept, baseline[concept], expanded[concept]])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
