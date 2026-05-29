# mlops_vehicletype_project

End-to-end MLOps pipeline for vehicle body-type classification on the
Vehicle-Type-10 dataset (1002 images, 10 classes). Covers DVC-versioned
data, MLflow experiment tracking + model registry, Kubeflow Pipelines
orchestration, KServe deployment, and Prometheus/Grafana monitoring.

## Repository layout
- `data/` — raw + processed images, versioned with DVC.
- `src/` — data prep, training scripts (baseline + K-fold), model factory.
- `config/` — YAML configs (MLflow tracking, etc.).
- `scripts/` — helper scripts (MLflow server launcher).
- `deployment/` — Dockerfiles + K8s manifests (in-cluster MLflow, KServe `InferenceService`, RBAC).
- `pipeline/` — Kubeflow Pipelines DAG definition.

## Quickstart
1. `pip install -r requirements.txt`
2. Place the raw Vehicle-Type-10 images under `data/raw/` (with `train/` and `test/` subfolders), then run `dvc repro prepare` to build train/val/test splits.
3. Start the MLflow server (see "Experiment tracking" below).
4. Train: `cd src && python train.py --data_dir ../data/processed`.


## Experiment tracking (MLflow)

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
 
## Deployment (Docker)

The trained model is served via a FastAPI app inside a Docker container that loads from the MLflow Model Registry by alias.

### Build
From the project root:
```powershell
docker build -f deployment/Dockerfile -t vehicle-classifier:dev .
```

### Run
The MLflow tracking server must be running on the host first.
```powershell
docker run --rm -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -e MODEL_URI=models:/vehicle-type-classifier@champion `
  vehicle-classifier:dev
```

### Endpoints
- `GET /health` → `{"status":"ok"}`
- `GET /metrics` — Prometheus metrics for request status, latency, predicted class, and confidence.
- `POST /predict` — multipart form upload of a single image; returns top-3 classes with confidences.

Example:
```powershell
curl.exe -F "file=@path/to/car.jpg" http://localhost:8000/predict
```

## Orchestration (Kubeflow Pipelines)

The training workflow runs as a parameterized DAG in Kubeflow Pipelines on a local minikube cluster. The pipeline:

```
data_prep_check  →  train_baseline  →  promote_champion  →  rollout_inference
```

(`train_kfold` is defined in [pipeline/pipeline.py](pipeline/pipeline.py) but disabled in the smoke pipeline because the `convnext_base_advanced` model doesn't fit alongside the rest of the stack in an 8 GiB minikube node. Re-enable on a bigger cluster.)

### Cluster prerequisites
- minikube ≥ 1.34 with the docker driver, at least 8 GiB memory (`minikube start --cpus 4 --memory 8192 --disk-size 40g --driver=docker`). 12 GiB+ recommended for `train_kfold`.
- Kubeflow Pipelines v2.16 installed (see [installation_kubeflowpiplines.txt](../week-10-kubeflow_and_local_deployment/installation_kubeflowpiplines.txt) — same procedure as week 10).
- `kfp` + `kfp-kubernetes` Python SDKs locally: `pip install "kfp>=2.0" kfp-kubernetes`.

### One-time cluster setup (KServe + in-cluster MLflow + data + RBAC)
From the project root:
```powershell
# 1. KServe + Knative Serving + Istio + cert-manager (see "Deployment (KServe)" below)

# 2. Application namespaces
kubectl apply -f deployment/k8s/00-namespaces.yaml

# 3. In-cluster MLflow (separate from the host MLflow used for local dev)
kubectl apply -f deployment/k8s/mlflow-pvc.yaml
kubectl apply -f deployment/k8s/mlflow-deployment.yaml
kubectl wait --for=condition=ready pod -l app=mlflow -n mlflow --timeout=180s

# 4. Data PVCs (one in vehicle ns for reference, one in kubeflow ns for pipeline pods)
kubectl apply -f deployment/k8s/data-pvc.yaml

# 5. Pipeline-runner RBAC (lets KFP pods patch the InferenceService)
kubectl apply -f deployment/k8s/pipeline-runner-rbac.yaml

# 6. Allow Knative to skip digest resolution for locally-built :dev images
kubectl patch configmap config-deployment -n knative-serving --type=merge `
  -p '{\"data\":{\"registries-skipping-tag-resolving\":\"kind.local,ko.local,dev.local,docker.io,index.docker.io\"}}'
```

### Seed the data PVC (one time per cluster)
KFP pipeline pods mount the `vehicle-data` PVC in the `kubeflow` namespace. Seed it from the host:
```powershell
# Spawn a temporary seeder pod
kubectl run vehicle-data-seeder -n kubeflow --image=busybox `
  --restart=Never --command -- sleep 3600 `
  --overrides='{ \"spec\":{\"containers\":[{\"name\":\"vehicle-data-seeder\",\"image\":\"busybox\",\"command\":[\"sleep\",\"3600\"],\"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]}], \"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"vehicle-data\"}}] } }'
kubectl wait --for=condition=ready pod/vehicle-data-seeder -n kubeflow --timeout=60s

# Copy data from host
kubectl cp data/processed vehicle-data-seeder:/data/processed --namespace=kubeflow

# Tear down
kubectl delete pod vehicle-data-seeder -n kubeflow
```

### Build images (into minikube's docker)
```powershell
docker build -f deployment/Dockerfile.train -t vehicle-trainer:dev .
docker build -f deployment/Dockerfile       -t vehicle-classifier:dev .
& "C:\Program Files\Kubernetes\Minikube\minikube.exe" image load vehicle-trainer:dev
& "C:\Program Files\Kubernetes\Minikube\minikube.exe" image load vehicle-classifier:dev
```
Note: if you rebuild an image with the same tag, also run `minikube image rm <tag>` first — `image load` won't overwrite a same-named image already cached in the cluster runtime.

### Submit the pipeline
KFP UI must be reachable. In a dedicated terminal:
```powershell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Then from the project root:
```powershell
python pipeline/pipeline.py
```
This compiles to `pipeline/pipeline.yaml` and submits a run via `kfp.Client(host='http://localhost:8080')`. Open the printed run URL to watch the DAG execute.

### What the pipeline does
1. `data_prep_check` — verifies the data PVC has `train/val/test` populated; fails fast otherwise.
2. `train_baseline` — runs [src/train.py](src/train.py) (1 epoch by default, small batch for smoke). Registers a new `vehicle-type-classifier` model version into in-cluster MLflow.
3. `promote_champion` — picks the best-`val_acc` version and assigns the `@champion` alias.
4. `rollout_inference` — patches the KServe `InferenceService` so its predictor pod restarts and loads the new champion. No-op if the `InferenceService` isn't deployed yet.

## Deployment (KServe)

KServe replaces raw `Deployment + Service + Ingress` with a single `InferenceService` resource that includes auto-scaling, traffic management, and a standardized predict URL — see [deployment/k8s/inferenceservice.yaml](deployment/k8s/inferenceservice.yaml).

### Install KServe + dependencies (one time)
```powershell
.\scripts\setup_kserve.ps1
```

### Deploy the InferenceService
```powershell
kubectl apply -f deployment/k8s/inferenceservice.yaml
kubectl get inferenceservice -n vehicle -w   # wait for READY=True
```

### Smoke-test /predict
The `InferenceService` is reachable cluster-internally at `http://vehicle-classifier.vehicle.svc.cluster.local`. From the host, port-forward the predictor pod and curl:
```powershell
$pod = kubectl get pods -n vehicle -o jsonpath='{.items[0].metadata.name}'
kubectl port-forward -n vehicle pod/$pod 8090:8000   # leave this running

# In a second terminal
curl.exe -s http://localhost:8090/health
curl.exe -F "file=@data/processed/val/sedan/<some-image>.jpg" http://localhost:8090/predict
```

### How a new model gets deployed
1. Re-run the pipeline (`python pipeline/pipeline.py`) → a new `vehicle-type-classifier` version is registered, the `@champion` alias moves to it.
2. The pipeline's `rollout_inference` component patches the `InferenceService` (mutates an env var), which triggers KServe to spawn a new predictor revision. The new revision loads `models:/vehicle-type-classifier@champion` on startup → serves the new model.
3. No manifest edit needed, no manual `kubectl rollout`. The `InferenceService` URL stays the same.

## Monitoring (Prometheus + Grafana)

The FastAPI app exposes Prometheus metrics at `/metrics`. Custom metrics:
- `vehicle_prediction_requests_total`
- `vehicle_prediction_classes_total`
- `vehicle_prediction_latency_seconds`
- `vehicle_prediction_confidence`

Install Prometheus + Grafana and apply the scraper:
```powershell
.\scripts\setup_monitoring.ps1
```

Open Grafana:
```powershell
kubectl port-forward -n monitoring svc/prom-grafana 3000:80
```

Get the Grafana password:
```powershell
kubectl get secret -n monitoring prom-grafana -o jsonpath="{.data.admin-password}" | % { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
```

Then open `http://localhost:3000` and log in as `admin`. Useful first panels: prediction count by class, request latency, error count, and confidence distribution.
