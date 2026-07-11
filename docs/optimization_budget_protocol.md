# Optimization Budget Evaluation Protocol

SEGS++ results should be reported as quality-versus-cost trade-offs, not as the best PSNR selected from unrelated hyperparameter runs. Each comparison must make the optimization budget explicit and hold one budget axis fixed while measuring the others.

## Fixed-budget comparisons

Evaluate SEGS++ and the baseline under each of the following matched budgets:

| Budget axis | What to hold fixed | What to report |
| --- | --- | --- |
| Iterations | Same optimizer iteration count and evaluation checkpoints. | PSNR, SSIM, MS-SSIM, LPIPS, edge similarity, saliency similarity, training time, Gaussian count. |
| Wall-clock training time | Same elapsed training-time cutoff on identical hardware. | Quality metrics at the cutoff, iteration reached, Gaussian count, GPU memory. |
| Gaussian count | Same maximum or post-pruning number of Gaussians, for example one million. | Quality metrics, model-file size, render FPS, training time needed to reach the count. |
| Model-file size | Same saved checkpoint or PLY size on disk. | Quality metrics, Gaussian count, render FPS, GPU memory. |
| GPU memory | Same peak CUDA memory limit or measured peak allocation. | Quality metrics, Gaussian count, training time, render FPS. |
| Rendering frame-rate | Same minimum render FPS at the target resolution, for example 30 FPS at 1080p. | Quality metrics, Gaussian count, model-file size, training time. |

## Pareto reporting

For every scene and method, record each run as a point containing:

- quality: PSNR, SSIM, MS-SSIM, LPIPS, edge similarity, and saliency similarity;
- optimization cost: iteration, wall-clock training time, and peak GPU memory;
- representation cost: number of Gaussians and model-file size;
- deployment cost: render FPS at the target resolution and hardware.

Plot Pareto curves for quality versus each cost axis. A run is Pareto-dominated if another run has equal or better quality with equal or lower cost and is strictly better on at least one axis. The strongest claims should be stated at fixed budgets, for example:

- At a fixed one-million-Gaussian budget, SEGS++ improves perceptual quality while preserving rendering speed.
- SEGS++ reaches the baseline LPIPS with 30% fewer Gaussians and 20% less training time.

## Minimum metadata for defensible claims

Include the dataset split, scene name, random seed, GPU model, render resolution, training command, evaluation command, and commit hash for every run. Do not compare runs that differ in hardware, resolution, train/test split, or evaluation script unless those differences are explicitly identified as separate ablations.
