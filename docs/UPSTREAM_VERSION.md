# Pinned upstream GraphDECO version

- **Official upstream repository:** https://github.com/graphdeco-inria/gaussian-splatting
- **Exact upstream commit SHA:** `d8856f60c5384cc1975439193bb627d77d917d77`
- **Date selected:** 2026-07-11
- **Selection basis:** This fork vendors the upstream SIBR viewer submodule at the commit above and retains the GraphDECO 3D Gaussian Splatting training layout while adding SEGS-specific training controls and demo asset tooling.

## Local changes to default optimization parameters

The demo training configs preserve upstream-style default 30,000-iteration optimization parameters for the baseline runs: standard L1 plus DSSIM objective, standard densification interval, opacity reset interval, densification window, pruning threshold, spherical-harmonic degree 3, black background, no antialiasing, and seed 0. SEGS-full runs intentionally add edge-and-saliency weighting, adaptive curriculum, and importance-aware densification/pruning as committed in this fork.

## Known differences from upstream

- SEGS methods are exposed through the `--method` argument in `train.py`.
- Baseline and SEGS-full demo jobs are driven by committed configs under `configs/demo_training/`.
- Demo packaging requires resolved configuration, runtime metadata, optimization budget, model card, load-test status, and deterministic tar.gz archives.
- `eggs` is exposed as the edge-only EGGS-style comparison, and `eggs_saliency` is exposed as the thesis method that adds saliency weighting to EGGS.

## Submodule commits used

```text
-d8856f60c5384cc1975439193bb627d77d917d77 SIBR_viewers
-26ce026ae9d3cfa56a103279b863a9f320c3e555 submodules/diff-gaussian-rasterization
-1272e21a282342e89537159e4bad508b19b34157 submodules/fused-ssim
-86710c2d4b46680c02301765dd79e465819c8f19 submodules/simple-knn
```
