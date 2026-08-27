# Large-scale subset experiments

The large-scale runner creates deterministic, class-balanced training subsets,
generates Grounding DINO annotations on two GPUs, trains VLG-CBM, and evaluates
the resulting checkpoint under the repository's NEC protocol.

## Experiments

| Code | Dataset | Images/class | Images | Config |
| --- | --- | ---: | ---: | --- |
| `p_large_1` | Places365 train | 300 | 109,500 | `configs/places365_p_large_1.json` |
| `p_large_2` | Places365 train | 600 | 219,000 | `configs/places365_p_large_2.json` |
| `i_large_1` | ImageNet-1K train | 100 | 100,000 | `configs/imagenet_i_large_1.json` |
| `i_large_2` | ImageNet-1K train | 200 | 200,000 | `configs/imagenet_i_large_2.json` |

Sampling ranks each path with SHA-256 independently inside every class. Rows are
then emitted by rank and class, so the smaller subset and its annotations are an
exact index prefix of the larger subset with the same seed. P-Large-2 therefore
resumes from P-Large-1 instead of annotating its first 109,500 images again.

## Run and monitor

Run an experiment in a detached session:

```bash
mkdir -p experiment_results/large_scale_runs/p_large_1
setsid -f ./scripts/run_large_scale_experiment.sh p_large_1 \
  > experiment_results/large_scale_runs/p_large_1/orchestrator.log 2>&1 \
  < /dev/null
```

Monitor the stage, GPU workers, and logs:

```bash
cat experiment_results/large_scale_runs/p_large_1/status.txt
cat experiment_results/large_scale_runs/p_large_1/orchestrator.pid
tail -f experiment_results/large_scale_runs/p_large_1/annotation_gpu0.log
tail -f experiment_results/large_scale_runs/p_large_1/annotation_gpu1.log
nvidia-smi
```

`ANNOTATION_BATCH_SIZE` and `ANNOTATION_WORKERS` can override the annotation
defaults. `VLG_CBM_PYTHON` can override the Python executable.

## Outputs

- Subset manifests: `datasets/subsets/`
- Resumable annotations: `annotations/places365_large_train/` or
  `annotations/imagenet_large_train/`
- Run state and logs: `experiment_results/large_scale_runs/<code>/`
- Concept scaling summary: `<run>/concept_scaling/summary.json`
- Per-concept support: `<run>/concept_scaling/concept_support.csv`
- NEC metrics: `<run>/anec.csv`
- Model checkpoint: `experiment_results/checkpoints/<experiment>/`

The concept report measures newly recovered baseline-zero concepts, baseline
low-frequency concepts that reach at least ten supporting images, and singleton
support rate as a clearly labeled noise proxy. It does not treat support counts
as direct false-positive ground truth.

ImageNet experiments require the training set at
`datasets/imagenet/ILSVRC/Data/CLS-LOC/train`. The current workspace only has the
50,000-image validation set, so the ImageNet runners intentionally fail before
launch if that training root is absent.

## Reconstructing dependencies on another server

Datasets, annotations, checkpoints, Grounding DINO weights, and local model
caches are intentionally excluded from Git. After cloning this repository, run:

```bash
./scripts/setup_groundingdino.sh
```

Then place `groundingdino_swinb_cogcoor.pth` inside `GroundingDINO/`. The BERT
encoder is loaded from `bert-base-uncased/` when that local directory exists;
otherwise the configured Hugging Face model identifier is used. It can also be
overridden with `GROUNDING_DINO_TEXT_ENCODER`.

If the Python interpreter is not at this machine's default path, set
`VLG_CBM_PYTHON=/path/to/python` before launching the runner.
