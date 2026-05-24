# mlops_vehicletype_project

End-to-end MLOps pipeline for vehicle body-type classification on the
Vehicle-Type-10 dataset (1002 images, 10 classes). Covers DVC-versioned
data, MLflow experiment tracking + model registry, and (later) Kubeflow
orchestration, KServe deployment, and Prometheus/Grafana monitoring.

## Repository layout
- `data/` — raw + processed images, versioned with DVC.
- `src/` — data prep, training scripts (baseline + K-fold), model factory.
- `config/` — YAML configs (MLflow tracking, etc.).
- `scripts/` — helper scripts (MLflow server launcher).

## Quickstart
1. `pip install -r requirements.txt`
2. `dvc pull` to fetch raw data, then `dvc repro prepare` to build train/val/test splits.
3. Start the MLflow server (see "Experiment tracking" below).
4. Train: `cd src && python train.py --data_dir ../data/processed`.


# Experiment tracking (MLflow)

All training runs are logged to a local MLflow tracking server.

### One-time setup
- Config lives in `config/mlflow.yaml` (tracking URI, experiment name, registered model name).
- The server uses SQLite for metadata (`mlflow.db`) and a local directory for artifacts (`mlruns_artifacts/`). Both are gitignored.

### Start the server
In a dedicated terminal, from the project root:
```powershell
./scripts/start_mlflow_server.ps1
```
UI: http://127.0.0.1:5000

### Run training
```powershell
cd src
python train.py --epochs 1 --data_dir ../data/processed --model_name swin_tiny
python train_model2.py --epochs 1 --folds 2 --data_dir ../data/processed/train
```

Each run logs:
- **Params**: all CLI arguments.
- **Metrics**: per-epoch `train_loss`, `train_acc`, `val_loss`, `val_acc` (baseline); per-fold `val_loss`, `val_acc` and run-level `cv_mean_acc`, `cv_std_acc` (K-fold).
- **Tags**: `script` (which trainer), `data_rev` (DVC md5 of the raw dataset — links the run to an exact data version).
- **Artifacts**: best model under `model/`, plus `model_meta/class_names.json` and `model_meta/meta.json`.
- **Registry**: a new version of `vehicle-type-classifier`.

### Promote a model for deployment
Tag the best version with the `champion` alias:
```powershell
python -c "from mlflow.tracking import MlflowClient; MlflowClient(tracking_uri='http://127.0.0.1:5000').set_registered_model_alias('vehicle-type-classifier', 'champion', version='<N>')"
```
Downstream services load by alias: `mlflow.pytorch.load_model("models:/vehicle-type-classifier@champion")`.
 