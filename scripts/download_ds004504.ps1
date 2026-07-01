param(
    [string]$Target = "data\ds004504"
)

New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null

if (Get-Command openneuro -ErrorAction SilentlyContinue) {
    openneuro download --dataset ds004504 --target-dir $Target
} elseif (python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openneuro') else 1)" 2>$null) {
    python -m openneuro download --dataset ds004504 --target-dir $Target
} elseif (Get-Command datalad -ErrorAction SilentlyContinue) {
    datalad install -r -s https://github.com/OpenNeuroDatasets/ds004504.git $Target
    datalad get "$Target\sub-*\eeg\*"
    datalad get "$Target\participants.tsv"
} else {
    Write-Error "Install openneuro-py or DataLad first. Example: python -m pip install openneuro-py"
}
