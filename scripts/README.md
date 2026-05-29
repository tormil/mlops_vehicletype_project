# Scripts

Run from the project root in PowerShell. Start Docker Desktop first.

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

## 1. Prepare Data

```powershell
dvc repro prepare
```

## 2. Set Up Kubeflow

```powershell
.\scripts\setup_minikube_kubeflow.ps1 -InstallMinikube -MemoryMb 12288 -BuildImages -SeedData
```

For an existing cluster:

```powershell
.\scripts\setup_minikube_kubeflow.ps1 -SkipKfpInstall -SkipSdkInstall -SkipMinikubeStart -MemoryMb 12288 -BuildImages -ReloadImages -SeedData
```

## 3. Set Up KServe

```powershell
.\scripts\setup_kserve.ps1
```

## 4. Run Training Pipeline

Terminal 1:

```powershell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe pipeline\pipeline.py
```

Open `http://localhost:8080`.

## 5. Deploy Model Service

```powershell
kubectl apply -f deployment/k8s/inferenceservice.yaml
kubectl get inferenceservice -n vehicle -w
```

## 6. Set Up Monitoring

```powershell
.\scripts\setup_monitoring.ps1
```

Open Grafana:

```powershell
kubectl port-forward -n monitoring svc/prom-grafana 3000:80
```

Open `http://localhost:3000`. Username: `admin`. The password command is in the main project README.

## 7. Test Prediction

```powershell
$pod = kubectl get pods -n vehicle -o jsonpath='{.items[0].metadata.name}'
kubectl port-forward -n vehicle pod/$pod 8090:8000
```

In another terminal:

```powershell
curl.exe -s http://localhost:8090/health
curl.exe -F "file=@data/processed/val/sedan/<image>.jpg" http://localhost:8090/predict
```
