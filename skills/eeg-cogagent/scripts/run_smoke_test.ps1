param(
    [string]$Config = "configs\ds004504_minimal.yaml",
    [int]$SubjectsLimit = 6,
    [string]$OutputDir = "results\smoke\ds004504_minimal"
)

$ErrorActionPreference = "Stop"

$ProjectPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $ProjectPython) { $ProjectPython } else { "python" }

& $Python -m pip install -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m eeg_cogagent.cli plan $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m eeg_cogagent.cli run $Config --subjects-limit $SubjectsLimit --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m eeg_cogagent.cli audit $Config --output-dir $OutputDir
exit $LASTEXITCODE
