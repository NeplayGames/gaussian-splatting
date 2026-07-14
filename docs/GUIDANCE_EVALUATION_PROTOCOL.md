# EGGS + Saliency Guidance Evaluation Protocol

## Research Question

Does saliency-aware image-space weighting improve EGGS/3DGS, and are gains due to meaningful spatial guidance or increased loss scale?

## Method Ladder

Run every scene with the same seed, train/test split, iteration budget, resolution, background setting, and rendering/metrics pipeline.

- `baseline`: standard 3DGS L1 + DSSIM objective.
- `eggs_paper`: paper-formula EGGS comparison, `w = 1 + beta * ||grad(image)||_p`, edge-only, raw/unnormalized.
- `eggs`: unnormalized EGGS-style edge weighting, `w = 1 + alpha * edge`.
- `saliency`: unnormalized saliency-only weighting, `w = 1 + beta * saliency`.
- `eggs_saliency`: unnormalized thesis method, `w = 1 + alpha * edge + beta * saliency`.
- `eggs_norm`: mean-normalized edge control, `w = raw / mean(raw)`.
- `saliency_norm`: mean-normalized saliency control.
- `eggs_saliency_norm`: mean-normalized thesis control.

The `eggs_paper` method is the clean comparison to the original EGGS formulation. The other unnormalized methods test the thesis extension route. The normalized methods test whether gains remain when the average loss scale is held near baseline.

## Saliency Backends

Run saliency-bearing methods with each configured backend:

- `BooleanMapApprox`
- `IntensityCenterSurround`

Report backend-specific results instead of averaging them silently.

## Metrics

Report full-image metrics:

- PSNR
- SSIM
- LPIPS

Report region-specific metrics from GT-derived masks:

- edge-region PSNR and MAE
- saliency-region PSNR and MAE
- mask coverage for both regions

The default protocol uses the top 10 percent of GT edge/saliency scores as each region.

## Runner

Edit `configs/guidance_protocol.json` so every `source` points to a prepared scene root, then run:

```powershell
E:\MiniConda\envs\segs-demo\python.exe tools\run_custom_methods.py --matrix configs\guidance_protocol.json --iterations 30000 --output-root demo_output\guidance_protocol --eggs-beta 0.2 --edge-p 1
```

For a short smoke test, lower `--iterations` and add `--no-region-metrics` if rendered outputs are not available yet.

The runner writes `guidance_ablation_results.csv` under the selected output root.

## Conclusion Rules

- If `eggs > baseline`, edge guidance helps, consistent with EGGS.
- If `eggs_saliency > eggs`, saliency improves beyond edge guidance.
- If `eggs_saliency_norm > eggs_norm`, saliency improves beyond edge guidance after controlling for global loss scale.
- If only unnormalized methods improve, gains may be partly explained by stronger effective L1 gradients.
- If region metrics improve but full-image metrics do not, claim localized perceptual/detail benefits, not global reconstruction superiority.
