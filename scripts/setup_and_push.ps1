param(
    [string]github_repo = 'https://github.com/R0hanG0yal/multi_agent_hive_system.git'
)

Write-Host "Creating venv..."
python -m venv .venv
Write-Host "Activating..."
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning "Activation script not found. Skipping activation."
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip
Write-Host "Installing minimal deps..."
pip install pydantic pytest numpy requests sentence-transformers
Write-Host "Run tests..."
pytest -q
Write-Host "Run demo..."
python .\scripts\demo_run.py
Write-Host "Committing & pushing to GitHub..."
git init
git add -A
git commit -m "Initial working prototype nCo-authored-by: Copilot 223556219+Copilot@users.noreply.github.com"
git remote remove origin 2>$null
git remote add origin $github_repo
git branch -M main
# We won't push automatically as it may require credentials
# git push -u origin main
Write-Host "DONE! You can now run 'git push -u origin main' if you have credentials ready."
