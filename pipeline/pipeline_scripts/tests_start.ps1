$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..")
$ResDir = Join-Path $ScriptDir "..\pipeline_result\tests_out"
New-Item -ItemType Directory -Force -Path $ResDir | Out-Null
Set-Location $RootDir

python -m pytest tests -q --junitxml=(Join-Path $ResDir "pytest.xml")
if ($LASTEXITCODE -ne 0) {
    Write-Host "---------- Failed tests found ----------"
    python -m pytest tests --collect-only -q
    exit 1
}
Write-Host "---------- All tests were successful ---------"
python -m pytest tests --collect-only -q
