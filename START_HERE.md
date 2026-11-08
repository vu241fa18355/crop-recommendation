# Start Here

1. Open this directory in VS Code (it contains `README.md`, `config.yaml`, `api`, and `src`).
2. Create and activate a local environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Generate the crop model and project outputs:

   ```powershell
   python .\src\crop_workflow.py all
   ```

4. Verify the bundle and tests:

   ```powershell
   python .\check_project.py
   python -m pytest .\tests -q
   ```

5. Start the API:

   ```powershell
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```

   Visit `http://127.0.0.1:8000/docs` to try a crop recommendation.

See [README.md](README.md) for Docker and Kubernetes instructions.
