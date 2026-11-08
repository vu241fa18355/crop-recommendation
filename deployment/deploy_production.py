"""Validate, build, and deploy the crop recommendation API."""

import argparse
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "crop-recommendation-api:1.0"


def run(command: list[str]) -> None:
    print(f"[INFO] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def validate_manifests() -> None:
    for name in ("deployment.yaml", "service.yaml"):
        path = ROOT / "k8s" / name
        with path.open(encoding="utf-8") as file:
            manifest = yaml.safe_load(file)
        if not isinstance(manifest, dict) or manifest.get("apiVersion") is None:
            raise ValueError(f"Invalid Kubernetes manifest: {path}")
        print(f"[PASS] Valid Kubernetes manifest: {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate bundle and Kubernetes manifests only.")
    parser.add_argument("--build-only", action="store_true", help="Build the Docker image without deploying.")
    args = parser.parse_args()

    model_path = ROOT / "deployment" / "model.pkl"
    metadata_path = ROOT / "deployment" / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run `python src/train.py` before deployment.")
    validate_manifests()
    print(f"[PASS] Deployment model bundle is present: {model_path.relative_to(ROOT)}")
    if args.check:
        return 0

    run(["docker", "build", "-t", IMAGE, "."])
    if args.build_only:
        return 0
    run(["kubectl", "apply", "-f", "k8s/deployment.yaml"])
    run(["kubectl", "apply", "-f", "k8s/service.yaml"])
    run(["kubectl", "rollout", "status", "deployment/crop-recommendation-api", "--timeout=180s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
