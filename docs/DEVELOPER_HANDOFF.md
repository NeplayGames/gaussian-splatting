# Developer handoff: complete the local one-command SEGS demonstration

This repository currently has the quick-start framework, but it is not finished because the dataset checksum and all four pretrained-model entries still contain placeholder values.

Send this complete instruction to the developer:

```text
You are working on:

https://github.com/NeplayGames/gaussian-splatting

Complete the local one-command SEGS demonstration that has already been added
to the repository.

Important restrictions:

- Do not add GitHub Actions.
- Do not add anything under .github/workflows/.
- Do not run training or evaluation automatically on GitHub.
- All testing, training, packaging, rendering, and evaluation must happen on a
  local or manually operated GPU computer.
- Do not implement EGGS.
- Keep the supported demo methods as baseline and segs_full.

The final required user experience is:

git clone --recursive https://github.com/NeplayGames/gaussian-splatting.git
cd gaussian-splatting
python -m tools.quickstart

That command must automatically download the official dataset and four
pretrained models, render them, calculate metrics, print a result table, and
generate demo_output/report.html without asking the user for dataset or model
paths.

==================================================
1. USE THE OFFICIAL DATASET
==================================================

Use the official GraphDECO-prepared Tanks & Temples and Deep Blending COLMAP
dataset archive.

Official information page:

https://github.com/graphdeco-inria/gaussian-splatting

Official direct archive:

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

The archive is linked by the official repository as:

T&T+DB COLMAP

Use these two scenes:

- truck from Tanks and Temples
- drjohnson from Deep Blending

Do not upload or commit the dataset archive into this repository.

==================================================
2. VERIFY THE DATASET ARCHIVE
==================================================

Download the archive completely on a local machine.

Record:

- Exact byte size
- SHA-256 checksum
- Final download URL
- Archive filename
- Actual extracted directory structure
- Required scene paths
- License and attribution information

Calculate SHA-256 using one of these commands.

Linux:

sha256sum tandt_db.zip

macOS:

shasum -a 256 tandt_db.zip

Windows PowerShell:

Get-FileHash .\tandt_db.zip -Algorithm SHA256

Do not copy a checksum from an unverified third-party page.

Inspect the archive after downloading and verify the real capitalization and
directory names for:

- truck
- drjohnson
- images
- sparse
- sparse/0
- cameras files
- images files
- points3D files

Update expected_extracted_files in the manifest to match the archive exactly.

==================================================
3. UPDATE THE DATASET MANIFEST ENTRY
==================================================

Edit:

configs/demo_manifest.json

Replace:

UNVERIFIED_REPLACE_AFTER_LOCAL_ASSET_AUDIT
size_bytes: -1

with the real dataset checksum and exact file size.

The dataset entry must include:

- name
- version
- URL
- SHA-256
- exact size_bytes
- archive filename
- official information page
- license or terms page
- attribution
- required scenes
- expected extracted files

The manifest parser must reject:

- UNVERIFIED values
- PENDING values
- Empty URLs
- Negative file sizes
- Invalid SHA-256 strings
- Missing required files

==================================================
4. TRAIN THE FOUR REQUIRED MODELS
==================================================

Create these four pretrained model packages:

1. truck / baseline
2. truck / segs_full
3. drjohnson / baseline
4. drjohnson / segs_full

Use:

- 30,000 iterations
- Seed 0
- Test split enabled
- The same repository commit for all four runs
- Fixed committed configurations
- The same image-resolution settings
- Identical baseline and SEGS evaluation settings

Record the exact training commands.

Do not use reduced-iteration models for the default pretrained demo.

==================================================
5. VERIFY THE METHOD CONFIGURATIONS
==================================================

For baseline:

- Standard 3DGS photometric loss
- Standard 3DGS densification
- No edge weighting
- No saliency weighting
- No curriculum
- No SEGS densification

For segs_full:

- Scheduled edge-and-saliency weighted loss
- SEGS importance-aware densification
- Seed 0
- The committed SEGS-full parameters
- 30,000 total iterations

Save the fully resolved configuration for every model.

==================================================
6. REQUIRED MODEL FILES
==================================================

Every model directory must contain:

- cfg_args
- cfg_args.json
- resolved_config.json
- runtime_metadata.json
- optimization_budget.json
- MODEL_CARD.md
- point_cloud/iteration_30000/point_cloud.ply
- Any exposure files required by render.py

runtime_metadata.json must include:

- Method
- Scene
- Seed
- Iterations
- Local Git commit
- Pinned upstream commit
- GPU name
- CUDA version
- PyTorch version
- Python version
- Training date
- Training duration

optimization_budget.json must include:

- Final Gaussian count
- Model-file size
- Peak GPU memory
- Training time
- Recorded rendering information when available

==================================================
7. CREATE MODEL CARDS
==================================================

Create MODEL_CARD.md inside every model package.

Include:

- Model name
- Method
- Scene
- Dataset source
- Seed
- Iterations
- Exact training command
- Local Git commit
- Upstream Git commit
- Configuration hash
- GPU
- CUDA version
- PyTorch version
- Training date
- Training time
- Known limitations
- Statement that the model is for demonstration and reproducibility

==================================================
8. PACKAGE THE MODELS
==================================================

Use or finish:

scripts/package_demo_models.py

The script must:

1. Accept a model directory.
2. Check every required file.
3. Reject incomplete models.
4. Confirm that iteration 30000 exists.
5. Confirm method, scene, and seed.
6. Calculate a configuration hash.
7. Create a deterministic archive.
8. Calculate SHA-256.
9. Record exact archive size.
10. Print JSON suitable for copying into demo_manifest.json.

Create these archive names:

truck_baseline_seed0_iter30000.tar.gz
truck_segs_full_seed0_iter30000.tar.gz
drjohnson_baseline_seed0_iter30000.tar.gz
drjohnson_segs_full_seed0_iter30000.tar.gz

==================================================
9. HOST THE MODEL ARCHIVES
==================================================

Upload the four model archives to stable public storage.

Acceptable options include:

- Zenodo
- Hugging Face
- University-hosted storage
- Manually uploaded GitHub Release assets

Do not commit the archives into the Git repository.

Do not use GitHub Actions to build or upload them.

Test every public URL from a separate computer or clean environment.

The URL must download the actual archive directly rather than an HTML page.

==================================================
10. COMPLETE ALL FOUR MODEL MANIFEST ENTRIES
==================================================

In:

configs/demo_manifest.json

Replace every occurrence of:

PENDING_STABLE_HOSTING_URL
PENDING_SHA256_AFTER_PACKAGING
PENDING_LOCAL_TRAINING_COMMIT
PENDING_UPSTREAM_COMMIT
PENDING_CONFIG_HASH
size_bytes: -1
pending-local-training

with verified values.

Every model entry must include:

- name
- final version
- method
- scene
- seed
- iteration
- direct download URL
- SHA-256
- exact size_bytes
- archive filename
- training commit
- upstream commit
- configuration hash
- license
- attribution
- required files
- expected extracted files

Do not merge a manifest containing PENDING or placeholder values.

==================================================
11. COMPLETE THE DOWNLOAD MANAGER
==================================================

The quick-start downloader must:

- Store downloads under ~/.cache/segs-demo
- Download to .part files
- Display progress
- Use timeouts
- Resume partial downloads when supported
- Verify exact file size
- Verify SHA-256
- Atomically rename verified files
- Reuse valid cached files
- Reject corrupt cached files
- Redownload corrupt files when online
- Fail clearly in offline mode when an asset is missing
- Prevent ZIP and TAR path traversal
- Safely extract only expected files
- Avoid extracting over already verified assets

Required cache layout:

~/.cache/segs-demo/
    downloads/
    datasets/
    models/
    manifests/

==================================================
12. COMPLETE DATASET VALIDATION
==================================================

After extraction, validate both scenes.

For every scene check:

- Scene directory exists
- Images directory exists
- Supported images are present
- Sparse COLMAP directory exists
- cameras.bin or cameras.txt exists
- images.bin or images.txt exists
- points3D.bin or points3D.txt exists
- Files are readable
- Scene is not partially extracted

Save:

demo_output/dataset_validation.json

Include:

- Dataset name
- Scene
- Source URL
- Archive SHA-256
- Archive size
- Image count
- Images path
- Sparse-model path
- Manifest version
- Validation timestamp

==================================================
13. COMPLETE MODEL VALIDATION
==================================================

Before rendering, validate each downloaded model.

Confirm:

- Correct scene
- Correct method
- Seed 0
- Iteration 30000
- Model package checksum
- Configuration hash
- Training commit
- Required files
- Point cloud exists and is nonempty
- Metadata JSON files parse successfully

Reject any model that does not match its manifest entry.

==================================================
14. COMPLETE THE QUICK-START PIPELINE
==================================================

The command:

python -m tools.quickstart

must perform this sequence:

1. Load configs/demo.yaml.
2. Load and validate configs/demo_manifest.json.
3. Check the environment.
4. Download and validate the dataset.
5. Extract and validate truck and drjohnson.
6. Download and validate all four pretrained models.
7. Render truck baseline.
8. Render truck segs_full.
9. Render drjohnson baseline.
10. Render drjohnson segs_full.
11. Calculate metrics.
12. Measure rendering performance.
13. Collect model and budget metadata.
14. Generate results.json.
15. Generate results.csv.
16. Print a terminal table.
17. Generate report.html.
18. Open the report unless --no-open is supplied.

Return exit code 0 only when all four evaluations succeed.

==================================================
15. RENDERING REQUIREMENTS
==================================================

For all four runs:

- Render iteration 30000
- Render test views only
- Use identical camera ordering
- Use identical resolution
- Use identical antialiasing settings
- Use identical exposure settings
- Save the exact command
- Capture stdout and stderr separately
- Stop on rendering failure
- Verify renders and ground-truth files exist

Store logs under:

demo_output/logs/truck/baseline/
demo_output/logs/truck/segs_full/
demo_output/logs/drjohnson/baseline/
demo_output/logs/drjohnson/segs_full/

==================================================
16. METRIC REQUIREMENTS
==================================================

Calculate:

- PSNR
- SSIM
- LPIPS

Use:

test/ours_30000

Generate:

demo_output/results.json
demo_output/results.csv

results.csv must contain exactly four evaluation rows:

- truck / baseline
- truck / segs_full
- drjohnson / baseline
- drjohnson / segs_full

Each row must contain:

- Dataset
- Scene
- Method
- Seed
- Iteration
- PSNR
- SSIM
- LPIPS
- Gaussian count
- Model size
- Training time
- Peak GPU memory
- Mean FPS
- Median FPS
- P95 frame time
- P99 frame time
- Repository commit
- Model-training commit
- Dataset checksum
- Model checksum

Do not hard-code metric values.

==================================================
17. RENDERING PERFORMANCE
==================================================

For each model:

1. Use the same ordered test-camera sequence.
2. Render 10 warm-up frames.
3. Synchronize CUDA.
4. Measure at least 100 frames.
5. Record individual frame times.
6. Calculate:
   - Mean FPS
   - Median FPS
   - Mean frame time
   - Median frame time
   - P95 frame time
   - P99 frame time

Save:

demo_output/render_performance.json

==================================================
18. HTML REPORT
==================================================

Generate:

demo_output/report.html

Include:

- SEGS demo description
- Repository commit
- Manifest version
- Environment information
- Dataset attribution
- Model metadata
- PSNR, SSIM, and LPIPS table
- Rendering-performance table
- Gaussian-count table
- Model-size table
- Baseline-versus-SEGS differences
- Ground-truth images
- Baseline renderings
- SEGS-full renderings
- Side-by-side comparison images
- Exact reproduction command
- Dataset checksum
- Model checksums
- Demonstration-results disclaimer

Use deterministic comparison-image selection.

Failure to open a browser must not fail the demo.

==================================================
19. FIX THE LOCAL ENVIRONMENT SETUP
==================================================

Review:

environment-demo.yml

Test the environment on an actual supported GPU computer.

Record:

- Operating system
- NVIDIA driver
- GPU model
- CUDA runtime
- PyTorch version
- Python version

Replace comments saying the environment is still untested.

Ensure all required packages are installed:

- PyTorch
- Torchvision
- PyYAML
- NumPy
- Pillow
- tqdm
- plyfile
- LPIPS dependency
- pytest
- diff-gaussian-rasterization
- simple-knn
- fused-ssim

==================================================
20. FIX demo.sh AND demo.ps1
==================================================

The scripts must run Python inside the created Conda environment.

For Linux/macOS, use a reliable approach such as:

conda run -n segs-demo python -m tools.quickstart "$@"

or correctly activate the environment before running Python.

For PowerShell, use the equivalent Conda command.

Do not create the environment and then accidentally run the system Python.

Do not silently ignore installation failures.

Remove patterns such as:

pip install ... || true

Installation failure must stop the script with a useful error.

==================================================
21. FIX .GITIGNORE
==================================================

Correct the accidental combined entry:

screenshotsdata/

Replace it with separate entries:

screenshots/
data/

Also keep:

datasets/
downloads/
checkpoints/
demo_output/
eval/
results/raw/
*.zip
*.tar
*.tar.gz
*.tar.zst
*.part

Do not ignore the manifest, configuration, documentation, or small test
fixtures.

==================================================
22. LOCAL UNIT TESTS
==================================================

Run locally:

python -m pytest
python -m compileall .

Tests must cover:

- Manifest validation
- Placeholder-value rejection
- SHA-256 verification
- Exact-size verification
- Safe ZIP extraction
- Safe TAR extraction
- Path traversal rejection
- Partial-download handling
- Cache reuse
- Corrupt-cache replacement
- Offline mode
- Missing dataset detection
- Missing model detection
- Model metadata mismatch
- Command construction
- Metrics loading
- CSV generation
- Terminal-table generation
- HTML-report generation
- EGGS rejection

Use only tiny synthetic fixtures for unit tests.

Do not download the full dataset during unit tests.

==================================================
23. LOCAL END-TO-END TEST
==================================================

Test on a clean supported GPU computer.

Run:

git clone --recursive https://github.com/NeplayGames/gaussian-splatting.git
cd gaussian-splatting

conda env create -f environment-demo.yml
conda activate segs-demo

git submodule update --init --recursive

python -m pip install submodules/diff-gaussian-rasterization
python -m pip install submodules/simple-knn
python -m pip install submodules/fused-ssim

python -m pytest
python -m compileall .

python -m tools.quickstart --check-only --no-open
python -m tools.quickstart --download-only --no-open
python -m tools.quickstart --no-open

Run it a second time:

python -m tools.quickstart --no-open

The second run must reuse the cache.

Then disconnect internet access and run:

python -m tools.quickstart --offline --no-open

Offline mode must complete successfully using the cached assets.

==================================================
24. VERIFY THE OUTPUTS
==================================================

Confirm these files exist and are nonempty:

demo_output/environment.json
demo_output/dataset_validation.json
demo_output/results.json
demo_output/results.csv
demo_output/render_performance.json
demo_output/report.html

Confirm the report contains comparison images.

Confirm the results contain all four scene/method combinations.

Confirm no large datasets, model archives, point clouds, or rendered outputs
were committed into Git.

==================================================
25. UPDATE DOCUMENTATION
==================================================

Update:

README.md
docs/QUICKSTART.md
docs/DEMO_ASSETS.md
docs/TROUBLESHOOTING.md
docs/LOCAL_TESTING.md

Document:

- Official dataset information page
- Official direct dataset URL
- Dataset size
- Dataset SHA-256
- Model URLs
- Model sizes
- Model checksums
- Required GPU
- Tested environment
- First-run behavior
- Cache location
- Offline mode
- Output files
- Expected runtime
- Troubleshooting
- Exact clean-clone commands

Do not leave statements saying URLs or checksums are pending.

Do not claim EGGS is implemented.

==================================================
26. DO NOT ADD GITHUB-SIDE EXECUTION
==================================================

Do not add:

- GitHub Actions
- CI workflows
- Scheduled jobs
- Cloud GPU jobs
- Automated release workflows
- Automated benchmark workflows

All validation must remain local.

==================================================
27. REQUIRED DELIVERABLES
==================================================

Provide:

1. Branch or pull-request link
2. Exact final Git commit
3. Dataset archive SHA-256
4. Dataset exact byte size
5. Four model download URLs
6. Four model SHA-256 checksums
7. Four model exact byte sizes
8. Four training commands
9. Four packaging commands
10. Model configuration hashes
11. Local unit-test output
12. Clean-machine environment report
13. First-run result
14. Cached second-run result
15. Offline-run result
16. demo_output/results.csv
17. demo_output/results.json
18. demo_output/render_performance.json
19. demo_output/report.html
20. List of remaining limitations

Do not call the feature finished until a different computer can perform a fresh
clone and successfully run:

python -m tools.quickstart

without manually providing dataset or model paths.
```
