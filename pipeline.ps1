$VENV_DIR = ".venv"

if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VENV_DIR
} else {
    Write-Host "Virtual environment already exists, skipping."
}

$VENV_PY = "$VENV_DIR\Scripts\python.exe"

Write-Host "Checking environment..."
& $VENV_PY env_check.py

Write-Host "`nActivating virtual environment..."
& "$VENV_DIR\Scripts\Activate.ps1"