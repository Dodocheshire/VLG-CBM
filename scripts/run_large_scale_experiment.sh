#!/usr/bin/env bash
set -Eeuo pipefail

experiment="${1:-p_large_1}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${VLG_CBM_PYTHON:-/home/pengcheng/miniconda3/envs/cbm/bin/python}"
annotation_batch_size="${ANNOTATION_BATCH_SIZE:-16}"
annotation_workers="${ANNOTATION_WORKERS:-4}"
export PYTHONPATH="${repo_root}/GroundingDINO${PYTHONPATH:+:${PYTHONPATH}}"

case "${experiment}" in
  p_large_1)
    dataset="places365"
    images_per_class=300
    class_count=365
    seed=42
    manifest_name="places365_p_large_1_train"
    annotation_dir="annotations/places365_large_train"
    baseline_annotation_dir="annotations/places365_val"
    config="configs/places365_p_large_1.json"
    checkpoint_root="experiment_results/checkpoints/places365_p_large_1"
    ;;
  p_large_2)
    dataset="places365"
    images_per_class=600
    class_count=365
    seed=42
    manifest_name="places365_p_large_2_train"
    annotation_dir="annotations/places365_large_train"
    baseline_annotation_dir="annotations/places365_val"
    config="configs/places365_p_large_2.json"
    checkpoint_root="experiment_results/checkpoints/places365_p_large_2"
    ;;
  i_large_1)
    dataset="imagenet"
    images_per_class=100
    class_count=1000
    seed=6885
    manifest_name="imagenet_i_large_1_train"
    annotation_dir="annotations/imagenet_large_train"
    baseline_annotation_dir="annotations/imagenet_val"
    config="configs/imagenet_i_large_1.json"
    checkpoint_root="experiment_results/checkpoints/imagenet_i_large_1"
    ;;
  i_large_2)
    dataset="imagenet"
    images_per_class=200
    class_count=1000
    seed=6885
    manifest_name="imagenet_i_large_2_train"
    annotation_dir="annotations/imagenet_large_train"
    baseline_annotation_dir="annotations/imagenet_val"
    config="configs/imagenet_i_large_2.json"
    checkpoint_root="experiment_results/checkpoints/imagenet_i_large_2"
    ;;
  *)
    echo "Unknown experiment: ${experiment}" >&2
    echo "Expected one of: p_large_1, p_large_2, i_large_1, i_large_2" >&2
    exit 2
    ;;
esac

cd "${repo_root}"
run_dir="experiment_results/large_scale_runs/${experiment}"
mkdir -p "${run_dir}" "${annotation_dir}" "${checkpoint_root}"
echo "$$" > "${run_dir}/orchestrator.pid"
status_file="${run_dir}/status.txt"
sample_count=$((images_per_class * class_count))
split_class=$(((class_count + 1) / 2))

on_error() {
  exit_code=$?
  for child_pid in "${gpu0_pid:-}" "${gpu1_pid:-}"; do
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill "${child_pid}" 2>/dev/null || true
    fi
  done
  echo "FAILED $(date --iso-8601=seconds) exit=${exit_code}" > "${status_file}"
  exit "${exit_code}"
}
trap on_error ERR

echo "SAMPLING $(date --iso-8601=seconds)" > "${status_file}"
"${python_bin}" scripts/sample_balanced_subset.py \
  --dataset "${dataset}" \
  --images-per-class "${images_per_class}" \
  --seed "${seed}" \
  --output-name "${manifest_name}"

echo "ANNOTATING $(date --iso-8601=seconds) expected=${sample_count}" > "${status_file}"
"${python_bin}" -m scripts.generate_annotations_multigpu \
  --dataset_name "${manifest_name}" \
  --output_dir "${annotation_dir}" \
  --device cuda:0 \
  --batch_size "${annotation_batch_size}" \
  --num_workers "${annotation_workers}" \
  --start_class_idx 0 \
  --end_class_idx "${split_class}" \
  > "${run_dir}/annotation_gpu0.log" 2>&1 &
gpu0_pid=$!

"${python_bin}" -m scripts.generate_annotations_multigpu \
  --dataset_name "${manifest_name}" \
  --output_dir "${annotation_dir}" \
  --device cuda:1 \
  --batch_size "${annotation_batch_size}" \
  --num_workers "${annotation_workers}" \
  --start_class_idx "${split_class}" \
  --end_class_idx "${class_count}" \
  > "${run_dir}/annotation_gpu1.log" 2>&1 &
gpu1_pid=$!

echo "${gpu0_pid}" > "${run_dir}/annotation_gpu0.pid"
echo "${gpu1_pid}" > "${run_dir}/annotation_gpu1.pid"
wait "${gpu0_pid}"
wait "${gpu1_pid}"

annotation_count=$(find "${annotation_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)
if (( annotation_count < sample_count )); then
  echo "Only ${annotation_count}/${sample_count} annotations were generated" >&2
  exit 1
fi

echo "ANALYZING $(date --iso-8601=seconds) annotations=${annotation_count}" > "${status_file}"
"${python_bin}" scripts/analyze_annotation_scaling.py \
  --concept-file "concept_files/${dataset}_filtered.txt" \
  --baseline-dir "${baseline_annotation_dir}" \
  --expanded-dir "${annotation_dir}" \
  --expanded-max-index "${sample_count}" \
  --output-dir "${run_dir}/concept_scaling" \
  > "${run_dir}/concept_scaling.log" 2>&1

echo "TRAINING $(date --iso-8601=seconds)" > "${status_file}"
"${python_bin}" train_cbm.py --config "${config}" > "${run_dir}/training.log" 2>&1

latest_checkpoint=$(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name "${dataset}_cbm_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${latest_checkpoint}" ]]; then
  echo "Training completed but no checkpoint was found in ${checkpoint_root}" >&2
  exit 1
fi
echo "EVALUATING $(date --iso-8601=seconds) checkpoint=${latest_checkpoint}" > "${status_file}"
"${python_bin}" sparse_evaluation.py \
  --load_path "${latest_checkpoint}" \
  --result_file "${run_dir}/anec.csv" \
  > "${run_dir}/sparse_evaluation.log" 2>&1

echo "COMPLETE $(date --iso-8601=seconds) checkpoint=${latest_checkpoint}" > "${status_file}"
