import json
import os
import yaml
import mlflow


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_mlflow_config(path=None):
    """Load tracking_uri, experiment_name, registered_model_name from YAML.

    Default path is resolved relative to the project root (the parent of src/),
    so the function works whether you launch Python from the project root or src/.
    """
    if path is None:
        path = os.path.join(_PROJECT_ROOT, "config", "mlflow.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_mlflow(cfg):
    """Point the MLflow client at the tracking server and experiment.

    Env vars take precedence over the YAML so the same scripts can run against
    the local host server (default) or an in-cluster server when invoked from
    a Kubeflow pipeline component.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg["experiment_name"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def get_dvc_data_rev(dvc_path="data/raw.dvc"):
    """Best-effort read of the DVC md5/hash so each run is tagged with its data version."""
    if not os.path.exists(dvc_path):
        return None
    with open(dvc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("hash:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("- md5:"):
                return line.split(":", 1)[1].strip()
    return None


def log_pytorch_best(model, registered_model_name, class_names, extra_meta,
                     tmp_dir="_tmp_artifacts"):
    """Log the best model to MLflow and create a new version under the registered model name.

    Also writes class_names.json + meta.json alongside the model so the deployment
    step has everything it needs to run inference.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(extra_meta, f, indent=2)
    mlflow.log_artifacts(tmp_dir, artifact_path="model_meta")

    return mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=registered_model_name,
    )
