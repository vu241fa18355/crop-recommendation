# Crop Recommendation Labs 1–13

Each lab is implemented for the crop dataset and can be run locally from the project root. Detailed notes are in the linked lab guides.

| Lab | Topic | Entry point / guide |
|---|---|---|
| 1 | Project setup and configuration | [README](../README.md) |
| 2 | Dataset loading and provenance | `python src/mlops_pipeline/01_load_data.py` |
| 3 | Dataset validation and baseline comparison | `python src/mlops_pipeline/02_validate_data.py` and `python src/mlops_pipeline/12_model_comparison.py` |
| 4 | Reproducible preprocessing and split | `python src/mlops_pipeline/03_preprocessing.py` |
| 5 | Crop model training | `python src/mlops_pipeline/05_train_registry.py` |
| 6 | Evaluation and macro-F1 quality gate | `python src/mlops_pipeline/06_model_evaluation.py` and `python src/mlops_pipeline/07_quality_gate.py` |
| 7 | Candidate/production model lifecycle | `python src/mlops_pipeline/08_automate_lifecycle.py` |
| 8 | Experiment tracking, model registry, and GitHub CI | `python src/mlops_pipeline/12_update_mlflow_registry.py`; [Lab 8 guide](LAB8_CI_REPORT.md) |
| 9 | Negative validation, failure handling, and automated tests | `python src/mlops_pipeline/10_failure_injection.py`; [Lab 9 guide](LAB9_FAILURE_HANDLING_REPORT.md) |
| 10 | Crop FastAPI single/batch serving | `python -m uvicorn api.main:app --port 8000`; open `/docs` |
| 11 | Docker and Kubernetes packaging | [Containerization guide](lab11_containerization_report.md) |
| 12 | Data/prediction/concept drift, request history, performance, retraining | `python src/mlops_pipeline/12_drift_analysis.py`; [Lab 12 guide](LAB12_MONITORING.md) |
| 13 | Self-hosted Windows CI, image smoke test, optional Docker Hub/Kubernetes release | [Self-hosted end-to-end guide](LAB13_END_TO_END.md) |

Run the complete local lifecycle and all tests:

```powershell
python src/crop_workflow.py all
python -m pytest tests -q
```

The original numbered lab entry points remain available; the monitoring and registry entries now perform crop-specific work instead of acting as no-op aliases. The optional concept-drift step correctly waits for labeled field feedback rather than inventing ground-truth outcomes.
