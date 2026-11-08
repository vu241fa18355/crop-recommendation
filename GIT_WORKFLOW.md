# Git Workflow

This project uses `https://github.com/vu241fa18355/crop-recommendation.git` on the `main` branch.

## Check the working tree

```powershell
git status --short --branch
git remote -v
```

## Publish a change

```powershell
git add .
git diff --cached --check
git commit -m "Describe the crop recommendation change"
git push origin main
```

Generated caches, local logs, virtual environments, and transient MLflow files are ignored. The crop source dataset and serving model bundle are intentionally part of the project.

## Pull changes

```powershell
git pull --ff-only origin main
```

Do not run Git commands from a parent directory that belongs to a different project. Run them from this project directory.
