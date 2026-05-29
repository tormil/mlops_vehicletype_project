# Scripts

Run all commands from the project root:

```powershell
cd C:\Users\merilipi\Documents\GitHub\mlops_vehicletype_project
Set-ExecutionPolicy -Scope Process Bypass
```

## Local MLflow

Start MLflow:

```powershell
.\scripts\start_mlflow_server.ps1
```

Open:

```text
http://127.0.0.1:5000
```

## Minikube + Kubeflow

Start Docker Desktop first.

For a first-time setup, run:

```powershell
.\scripts\setup_minikube_kubeflow.ps1 -InstallMinikube -MemoryMb 12288 -BuildImages -SeedData
```

If Kubeflow is already installed and you only need to continue project setup, run:

```powershell
.\scripts\setup_minikube_kubeflow.ps1 -SkipKfpInstall -SkipSdkInstall -SkipMinikubeStart -MemoryMb 12288 -BuildImages -SeedData
```

Open Kubeflow Pipelines:

```powershell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Then visit:

```text
http://localhost:8080
```

Submit the pipeline:

```powershell
.\.venv\Scripts\python.exe pipeline\pipeline.py
```

If the Kubeflow manifest download times out, rerun with:

```powershell
.\scripts\setup_minikube_kubeflow.ps1 -InstallMinikube -UseLocalKfpManifests -MemoryMb 12288 -BuildImages -SeedData
```
