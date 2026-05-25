"""One-off migration — APPLIED 2026-05-25. Do not re-run.
Re-logs the @champion model under the new artifact proxy. ...
"""

"""One-off: re-log the current @champion model so its artifacts live under the
new --serve-artifacts proxy (mlflow-artifacts:/...), then move the @champion
alias to the new version.

Run from the mlops_vehicletype_project directory with the proxy-enabled MLflow
server already running on http://127.0.0.1:5000.
"""
import json
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "vehicle-type-classification"
MODEL_NAME = "vehicle-type-classifier"
ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

old_mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
print(f"Current @{ALIAS}: version {old_mv.version} (run_id={old_mv.run_id})")

model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

with tempfile.TemporaryDirectory() as tmp:
    class_names_path = mlflow.artifacts.download_artifacts(
        run_id=old_mv.run_id,
        artifact_path="model_meta/class_names.json",
        dst_path=tmp,
    )
    with open(class_names_path) as f:
        class_names = json.load(f)
    try:
        meta_path = mlflow.artifacts.download_artifacts(
            run_id=old_mv.run_id,
            artifact_path="model_meta/meta.json",
            dst_path=tmp,
        )
        with open(meta_path) as f:
            extra_meta = json.load(f)
    except Exception:
        extra_meta = {"reregistered_from_version": old_mv.version}

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"reregister-v{old_mv.version}-via-proxy") as run:
    mlflow.set_tag("reregistered_from_version", old_mv.version)
    mlflow.set_tag("reregistered_from_run", old_mv.run_id)

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "class_names.json"), "w") as f:
            json.dump(class_names, f)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(extra_meta, f, indent=2)
        mlflow.log_artifacts(tmp, artifact_path="model_meta")

    info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )
    new_run_id = run.info.run_id

new_version = client.search_model_versions(f"run_id='{new_run_id}'")[0].version
print(f"Logged new model version: {new_version}")

client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)
print(f"Moved @{ALIAS} from v{old_mv.version} -> v{new_version}")

mv = client.get_model_version(MODEL_NAME, new_version)
print(f"New source URI: {mv.source}")
