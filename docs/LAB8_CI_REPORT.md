# Continuous Integration

The GitHub Actions workflows use GitHub-hosted Ubuntu runners and Python 3.11. The main workflow installs `requirements.txt`, runs the crop training pipeline and project preflight, then executes the API tests.

The validation workflow checks the dataset and invalid-data paths. The scheduled monitoring workflow writes a crop-only feature-range report. The runner-check workflow validates both the crop CSV and Kubernetes manifest structure without requiring a cluster.

Run the equivalent checks locally:

```powershell
python -m pip install -r requirements.txt
python .\src\crop_workflow.py all
python .\check_project.py
python -m pytest .\tests -q
```
