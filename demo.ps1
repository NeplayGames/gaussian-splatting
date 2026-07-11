param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$envName = if ($env:SEGS_DEMO_ENV) { $env:SEGS_DEMO_ENV } else { "segs-demo" }
$conda = Get-Command mamba -ErrorAction SilentlyContinue
if (-not $conda) { $conda = Get-Command conda -ErrorAction SilentlyContinue }
if ($conda) { & $conda.Source env create -n $envName -f environment-demo.yml 2>$null }
git submodule update --init --recursive
python -m pip install submodules/diff-gaussian-rasterization submodules/simple-knn
python -m tools.quickstart @Args
