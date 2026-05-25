"""One-off migration — APPLIED 2026-05-25. Do not re-run.
Re-logs the @champion model under the new artifact proxy. ...
"""

"""One-off: rename the legacy file:// experiment out of the way, then move
the proxy experiment into the canonical name. Run with the MLflow server up.
"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

OLD = "vehicle-type-classification"
LEGACY = "vehicle-type-classification-legacy"
PROXY = "vehicle-type-classification-proxy"

old = client.get_experiment_by_name(OLD)
proxy = client.get_experiment_by_name(PROXY)

if old is None and proxy is None:
    raise SystemExit("Neither source experiment found; nothing to do.")

if old is not None:
    client.rename_experiment(old.experiment_id, LEGACY)
    print(f"Renamed experiment {old.experiment_id}: '{OLD}' -> '{LEGACY}'  "
          f"(artifact_location={old.artifact_location})")
else:
    print(f"'{OLD}' not present; skipping legacy rename.")

if proxy is not None:
    client.rename_experiment(proxy.experiment_id, OLD)
    print(f"Renamed experiment {proxy.experiment_id}: '{PROXY}' -> '{OLD}'  "
          f"(artifact_location={proxy.artifact_location})")
else:
    print(f"'{PROXY}' not present; nothing to promote.")

for e in client.search_experiments():
    print(f"  id={e.experiment_id}  name={e.name!r}  artifact_location={e.artifact_location}")
