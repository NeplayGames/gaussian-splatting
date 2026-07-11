# Demo model training report

Status: **not run in this non-GPU automation container**.

Task 2 requires four 30,000-iteration trainings on a manually operated local GPU computer. This repository update adds the committed configs and driver needed for that run, but no trained checkpoint or GPU-derived metadata is included in Git.

Required command on the GPU host:

```bash
python scripts/train_demo_models.py --dataset-root <verified dataset root> --output-root demo_model_training
```

Before final training, commit the code/configuration and record:

```bash
git rev-parse HEAD
git status --short
git submodule status --recursive
```
