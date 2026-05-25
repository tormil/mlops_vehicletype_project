# Run from the mlops_vehicletype_project directory.
# Starts a local MLflow tracking server with SQLite backend + local artifact store.
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns_artifacts `
  --host 127.0.0.1 --port 5000
