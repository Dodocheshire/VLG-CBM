#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${repo_root}/experiment_results/large_scale_runs/places_scaling_completion"
mkdir -p "${run_dir}"
echo "$$" > "${run_dir}/orchestrator.pid"

on_error() {
  exit_code=$?
  echo "FAILED $(date --iso-8601=seconds) exit=${exit_code}" > "${run_dir}/status.txt"
  exit "${exit_code}"
}
trap on_error ERR

cd "${repo_root}"
echo "RUNNING_P_LARGE_2 $(date --iso-8601=seconds)" > "${run_dir}/status.txt"
./scripts/run_large_scale_experiment.sh p_large_2

# P-Large-1 was trained before the filtered-overlap crop fix. Re-run its cheap
# training/evaluation stages so both scale points use identical CBL semantics;
# the existing Grounding DINO annotations are resumed without recomputation.
echo "RERUNNING_P_LARGE_1 $(date --iso-8601=seconds)" > "${run_dir}/status.txt"
./scripts/run_large_scale_experiment.sh p_large_1

echo "COMPLETE $(date --iso-8601=seconds)" > "${run_dir}/status.txt"
