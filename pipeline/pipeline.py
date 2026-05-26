"""KFP v2 pipeline for the vehicle-type-classification project.

Five components:
  data_prep_check -> verifies the data PVC is populated
  train_baseline  -> shells out to src/train.py (registers a model version)
  train_kfold     -> shells out to src/train_model2.py (registers another version)
  promote_champion -> picks the best by val_acc and sets the @champion alias
  rollout_inference -> patches the KServe InferenceService (no-op if not deployed yet)

Submit with:
    python pipeline/pipeline.py
"""
from kfp import dsl, compiler, kubernetes
import kfp


MLFLOW_URI = "http://mlflow-service.mlflow.svc.cluster.local:5000"
TRAINER_IMAGE = "vehicle-trainer:dev"
DATA_PVC = "vehicle-data"
DATA_MOUNT = "/data"


@dsl.component(base_image=TRAINER_IMAGE)
def data_prep_check() -> str:
    import os
    expected = ["/data/processed/train", "/data/processed/val", "/data/processed/test"]
    for p in expected:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing required directory: {p}")
    classes = os.listdir("/data/processed/train")
    if len(classes) < 10:
        raise ValueError(f"Expected >=10 train classes, got {len(classes)}: {classes}")
    print(f"data PVC OK: {len(classes)} classes; train/val/test all present")
    return "ok"


@dsl.component(base_image=TRAINER_IMAGE)
def train_baseline(prep_status: str, epochs: int, batch_size: int) -> str:
    import subprocess, sys
    cmd = [
        "python", "/app/src/train.py",
        "--data_dir", "/data/processed",
        "--model_name", "swin_tiny",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited {proc.returncode}")
    return "baseline-done"


@dsl.component(base_image=TRAINER_IMAGE)
def train_kfold(prep_status: str, epochs: int, folds: int) -> str:
    import subprocess, sys
    cmd = [
        "python", "/app/src/train_model2.py",
        "--data_dir", "/data/processed/train",
        "--epochs", str(epochs),
        "--folds", str(folds),
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"train_model2.py exited {proc.returncode}")
    return "kfold-done"


@dsl.component(base_image=TRAINER_IMAGE)
def promote_champion(baseline_status: str, kfold_status: str) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri("http://mlflow-service.mlflow.svc.cluster.local:5000")
    client = MlflowClient()
    versions = client.search_model_versions("name='vehicle-type-classifier'")
    if not versions:
        raise RuntimeError("No model versions registered yet")
    best = None
    best_acc = -1.0
    for v in versions:
        run = client.get_run(v.run_id)
        # train.py logs val_acc; train_model2.py logs cv_mean_acc on parent run
        acc = float(run.data.metrics.get("val_acc", run.data.metrics.get("cv_mean_acc", -1)))
        print(f"  v{v.version} run={v.run_id[:8]} acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best = v
    client.set_registered_model_alias("vehicle-type-classifier", "champion", best.version)
    print(f"Promoted v{best.version} to @champion (acc={best_acc:.4f})")
    return f"v{best.version}"


@dsl.component(base_image=TRAINER_IMAGE, packages_to_install=["kubernetes==30.1.0"])
def rollout_inference(promoted_version: str) -> str:
    """Patch the InferenceService so KServe restarts the predictor pod and
    the new @champion model gets loaded. No-op if the InferenceService
    isn't deployed yet."""
    from kubernetes import client, config
    import time
    config.load_incluster_config()
    api = client.CustomObjectsApi()
    body = {
        "spec": {
            "predictor": {
                "containers": [
                    {
                        "name": "kserve-container",
                        "env": [
                            {"name": "ROLLOUT_TIMESTAMP", "value": str(int(time.time()))},
                            {"name": "PROMOTED_VERSION", "value": promoted_version},
                        ],
                    }
                ]
            }
        }
    }
    try:
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace="vehicle",
            plural="inferenceservices",
            name="vehicle-classifier",
            body=body,
        )
        print(f"Triggered InferenceService rollout for {promoted_version}")
        return "rolled-out"
    except client.exceptions.ApiException as e:
        if e.status == 404:
            print("InferenceService 'vehicle-classifier' not deployed yet — skipping rollout")
            return "skipped"
        raise


@dsl.pipeline(name="vehicle-classifier-pipeline")
def pipeline(epochs: int = 1, batch_size: int = 8):
    """Smoke-test pipeline: prep -> train_baseline -> promote_champion -> rollout.

    The train_kfold component is defined above for the full course run but is
    disabled in this smoke pipeline because the convnext_base_advanced model
    doesn't fit alongside KFP+KServe+Istio in an 8 GiB minikube node. Once the
    cluster has more memory (12 GiB+), re-enable it as a sequential step after
    train_baseline.
    """
    prep = data_prep_check()
    kubernetes.mount_pvc(prep, pvc_name=DATA_PVC, mount_path=DATA_MOUNT)
    prep.set_caching_options(False)

    base = train_baseline(prep_status=prep.output, epochs=epochs, batch_size=batch_size)
    kubernetes.mount_pvc(base, pvc_name=DATA_PVC, mount_path=DATA_MOUNT)
    base.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
    base.set_memory_limit("6Gi")
    base.set_cpu_limit("3")
    base.set_caching_options(False)

    promote = promote_champion(baseline_status=base.output, kfold_status="skipped")
    promote.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
    promote.set_caching_options(False)

    rollout = rollout_inference(promoted_version=promote.output)
    rollout.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, package_path="pipeline/pipeline.yaml")
    print("Compiled to pipeline/pipeline.yaml")

    client = kfp.Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_func(
        pipeline,
        arguments={"epochs": 1, "batch_size": 8},
        experiment_name="vehicle-classifier",
        run_name="smoke-run",
        enable_caching=False,
    )
    print(f"Submitted run: {run.run_id}")
    print(f"KFP UI: http://localhost:8080/#/runs/details/{run.run_id}")
