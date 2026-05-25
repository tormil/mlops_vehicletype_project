# Run from the mlops_vehicletype_project directory.
# Starts a local MLflow tracking server with SQLite backend + local artifact store.
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination ./mlruns_artifacts `
  --serve-artifacts `
  --host 0.0.0.0 --port 5000 `
  --allowed-hosts localhost,127.0.0.1,host.docker.internal,localhost:5000,127.0.0.1:5000,host.docker.internal:5000 `
  --cors-allowed-origins "http://localhost:5000,http://127.0.0.1:5000,http://host.docker.internal:5000"
