# Lab 1 — Project Setup

Create a Python virtual environment, install `requirements.txt`, and review `config.yaml`. The project root and configured paths are resolved relative to the source files; no developer-specific absolute paths are required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/crop_workflow.py validate
```
