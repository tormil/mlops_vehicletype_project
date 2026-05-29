<#
.SYNOPSIS
Sets up Minikube and Kubeflow Pipelines for this project.

.DESCRIPTION
This script starts a local Minikube cluster, installs Kubeflow Pipelines,
applies the project Kubernetes resources for MLflow/data/RBAC, and can
optionally build/load the project container images and seed the KFP data PVC.

.EXAMPLE
.\scripts\setup_minikube_kubeflow.ps1

.EXAMPLE
.\scripts\setup_minikube_kubeflow.ps1 -BuildImages -SeedData

.EXAMPLE
.\scripts\setup_minikube_kubeflow.ps1 -MemoryMb 12288 -BuildImages -SeedData -PortForward
#>

[CmdletBinding()]
param(
    [int]$Cpus = 4,
    [int]$MemoryMb = 8192,
    [string]$DiskSize = "40g",
    [ValidateSet("docker", "hyperv", "virtualbox", "podman", "wsl2")]
    [string]$Driver = "docker",
    [string]$KfpVersion = "2.16.0",
    [string]$KustomizeRemoteTimeout = "180s",

    [switch]$InstallMinikube,
    [switch]$SkipSdkInstall,
    [switch]$SkipMinikubeStart,
    [switch]$SkipKfpInstall,
    [switch]$SkipProjectResources,
    [switch]$UseLocalKfpManifests,

    [switch]$BuildImages,
    [switch]$ReloadImages,
    [switch]$SeedData,
    [switch]$PortForward
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$KnownPaths = @()
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    foreach ($path in $KnownPaths) {
        if (Test-Path $path) {
            return (Resolve-Path $path).Path
        }
    }

    return $null
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$ArgumentList
    )

    Write-Host "> $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkGray
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($ArgumentList -join ' ')"
    }
}

function Resolve-ProjectPython {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }

    $python = Resolve-Tool "python"
    if (-not $python) {
        throw "Python was not found. Create/activate a venv first, or install Python and rerun this script."
    }

    return $python
}

function Wait-Deployments {
    param(
        [Parameter(Mandatory = $true)][string]$Namespace,
        [Parameter(Mandatory = $true)][string]$Kubectl,
        [string[]]$Skip = @()
    )

    $deployments = & $Kubectl get deployments -n $Namespace -o name
    if ($LASTEXITCODE -ne 0) {
        throw "Could not list deployments in namespace '$Namespace'."
    }

    foreach ($deployment in $deployments) {
        $name = ($deployment -replace '^deployment\.apps/', '' -replace '^deployment/', '')
        if ($Skip -contains $name) {
            Write-Host "Skipping rollout wait for optional deployment: $name"
            continue
        }

        Invoke-Checked $Kubectl "rollout" "status" $deployment "-n" $Namespace "--timeout=600s"
    }
}

function Disable-OptionalKfpProxyAgent {
    param(
        [Parameter(Mandatory = $true)][string]$Kubectl
    )

    $exists = & $Kubectl get deployment proxy-agent -n kubeflow --ignore-not-found=true -o name
    if ($LASTEXITCODE -ne 0) {
        throw "Could not check proxy-agent deployment status."
    }

    if ($exists) {
        Write-Host "Scaling optional KFP proxy-agent to 0 replicas for local Minikube."
        Invoke-Checked $Kubectl "scale" "deployment/proxy-agent" "-n" "kubeflow" "--replicas=0"
    }
}

function Patch-KfpManifestRemoteTimeouts {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestRoot
    )

    $kustomizeRoot = Join-Path $ManifestRoot "manifests\kustomize"
    if (-not (Test-Path $kustomizeRoot)) {
        throw "Kubeflow kustomize manifest root was not found: $kustomizeRoot"
    }

    $pattern = 'https://github\.com/[^\s"'']+\?ref=[^&\s"'']+(?=($|[\s"'']))'
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $patchedCount = 0

    $files = Get-ChildItem -Path $kustomizeRoot -Recurse -File -Include "kustomization.yaml", "kustomization.yml"
    foreach ($file in $files) {
        $text = [System.IO.File]::ReadAllText($file.FullName)
        $updated = [regex]::Replace($text, $pattern, {
            param($match)
            "$($match.Value)&timeout=$KustomizeRemoteTimeout"
        })

        if ($updated -ne $text) {
            [System.IO.File]::WriteAllText($file.FullName, $updated, $encoding)
            $patchedCount++
        }
    }

    if ($patchedCount -gt 0) {
        Write-Host "Added Kustomize remote timeout to $patchedCount nested manifest file(s)."
    }
}

function Get-KfpManifestRoot {
    $gitKnownPaths = @(
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files\Git\bin\git.exe"
    )
    $GitCmd = Resolve-Tool "git" -KnownPaths $gitKnownPaths
    if (-not $GitCmd) {
        throw "Git was not found. Install Git or rerun without -UseLocalKfpManifests."
    }

    $tmpRoot = Join-Path $RepoRoot ".tmp"
    $manifestRoot = Join-Path $tmpRoot "kubeflow-pipelines-$KfpVersion"
    $gitDir = Join-Path $manifestRoot ".git"

    if (Test-Path $gitDir) {
        Write-Host "Refreshing existing Kubeflow Pipelines manifests at $manifestRoot"
        Invoke-Checked $GitCmd "-C" $manifestRoot "fetch" "--depth" "1" "origin" $KfpVersion
        Invoke-Checked $GitCmd "-C" $manifestRoot "checkout" "FETCH_HEAD"
        Patch-KfpManifestRemoteTimeouts -ManifestRoot $manifestRoot
        return $manifestRoot
    }

    if (Test-Path $manifestRoot) {
        throw "Manifest cache path exists but is not a Git checkout: $manifestRoot. Move it aside or remove it, then rerun."
    }

    New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
    Write-Host "Cloning Kubeflow Pipelines manifests into $manifestRoot"
    Invoke-Checked $GitCmd "clone" "--depth" "1" "--branch" $KfpVersion "https://github.com/kubeflow/pipelines.git" $manifestRoot
    Patch-KfpManifestRemoteTimeouts -ManifestRoot $manifestRoot
    return $manifestRoot
}

function Invoke-KubectlApplyKustomize {
    param(
        [Parameter(Mandatory = $true)][string]$Kubectl,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$LocalPath
    )

    if (-not $UseLocalKfpManifests) {
        $remoteRef = "${RemotePath}?ref=${KfpVersion}&timeout=${KustomizeRemoteTimeout}"
        Write-Host "Applying remote Kustomize manifest: $remoteRef"
        & $Kubectl apply -k $remoteRef
        if ($LASTEXITCODE -eq 0) {
            return
        }

        Write-Warning "Remote Kustomize apply failed. Falling back to a local shallow clone of kubeflow/pipelines."
    }
    else {
        Write-Host "Using local Kubeflow Pipelines manifests because -UseLocalKfpManifests was supplied."
    }

    $manifestRoot = Get-KfpManifestRoot
    $fullLocalPath = Join-Path $manifestRoot $LocalPath
    if (-not (Test-Path $fullLocalPath)) {
        throw "Kubeflow manifest path was not found: $fullLocalPath"
    }

    Invoke-Checked $Kubectl "apply" "-k" $fullLocalPath
}

function Start-MinikubeIfNeeded {
    param(
        [Parameter(Mandatory = $true)][string]$Minikube
    )

    if ($SkipMinikubeStart) {
        Write-Host "Skipping Minikube start because -SkipMinikubeStart was supplied."
        return
    }

    Write-Step "Checking Minikube status"
    $statusOutput = & $Minikube status --format "{{.Host}}" 2>$null
    $statusCode = $LASTEXITCODE

    if ($statusCode -eq 0 -and (($statusOutput -join "`n") -match "Running")) {
        Write-Host "Minikube is already running."
    }
    else {
        Invoke-Checked $Minikube "start" "--cpus" "$Cpus" "--memory" "$MemoryMb" "--disk-size" $DiskSize "--driver" $Driver
    }

    Invoke-Checked $Minikube "update-context"
}

function Install-KubeflowPipelines {
    param(
        [Parameter(Mandatory = $true)][string]$Kubectl
    )

    if ($SkipKfpInstall) {
        Write-Host "Skipping Kubeflow Pipelines install because -SkipKfpInstall was supplied."
        return
    }

    Write-Step "Installing Kubeflow Pipelines $KfpVersion"
    $clusterScopedRemote = "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources"
    $devEnvRemote = "github.com/kubeflow/pipelines/manifests/kustomize/env/dev"
    $clusterScopedLocal = "manifests/kustomize/cluster-scoped-resources"
    $devEnvLocal = "manifests/kustomize/env/dev"

    Invoke-KubectlApplyKustomize -Kubectl $Kubectl -RemotePath $clusterScopedRemote -LocalPath $clusterScopedLocal
    Invoke-Checked $Kubectl "wait" "--for=condition=established" "--timeout=60s" "crd/applications.app.k8s.io"
    Invoke-KubectlApplyKustomize -Kubectl $Kubectl -RemotePath $devEnvRemote -LocalPath $devEnvLocal
    Disable-OptionalKfpProxyAgent -Kubectl $Kubectl

    Write-Step "Waiting for Kubeflow Pipelines deployments"
    Wait-Deployments -Namespace "kubeflow" -Kubectl $Kubectl -Skip @("proxy-agent")
}

function Apply-ProjectResources {
    param(
        [Parameter(Mandatory = $true)][string]$Kubectl
    )

    if ($SkipProjectResources) {
        Write-Host "Skipping project Kubernetes resources because -SkipProjectResources was supplied."
        return
    }

    Write-Step "Applying project Kubernetes resources"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/00-namespaces.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/mlflow-pvc.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/mlflow-deployment.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/data-pvc.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/pipeline-runner-rbac.yaml"

    Write-Step "Waiting for in-cluster MLflow"
    Invoke-Checked $Kubectl "wait" "--for=condition=ready" "pod" "-l" "app=mlflow" "-n" "mlflow" "--timeout=180s"
}

function Remove-MinikubeImageIfPresent {
    param(
        [Parameter(Mandatory = $true)][string]$Minikube,
        [Parameter(Mandatory = $true)][string]$Image
    )

    Write-Host "Removing cached Minikube image if present: $Image"
    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = 0
    try {
        $ErrorActionPreference = "Continue"
        & $Minikube image rm $Image *> $null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        Write-Host "Could not remove cached image, continuing with image load."
    }
}

function Build-And-LoadImages {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [Parameter(Mandatory = $true)][string]$Minikube
    )

    if (-not $BuildImages) {
        Write-Host "Skipping image build/load. Rerun with -BuildImages when you need fresh images."
        return
    }

    Write-Step "Building Docker images"
    Invoke-Checked $Docker "build" "-f" "deployment/Dockerfile.train" "-t" "vehicle-trainer:dev" "."
    Invoke-Checked $Docker "build" "-f" "deployment/Dockerfile" "-t" "vehicle-classifier:dev" "."

    if ($ReloadImages) {
        Write-Step "Removing cached Minikube images"
        Remove-MinikubeImageIfPresent -Minikube $Minikube -Image "vehicle-trainer:dev"
        Remove-MinikubeImageIfPresent -Minikube $Minikube -Image "vehicle-classifier:dev"
    }

    Write-Step "Loading images into Minikube"
    Invoke-Checked $Minikube "image" "load" "vehicle-trainer:dev"
    Invoke-Checked $Minikube "image" "load" "vehicle-classifier:dev"
}

function Seed-DataPvc {
    param(
        [Parameter(Mandatory = $true)][string]$Kubectl
    )

    if (-not $SeedData) {
        Write-Host "Skipping data PVC seed. Rerun with -SeedData after data/processed exists."
        return
    }

    if (-not (Test-Path "data/processed")) {
        throw "data/processed does not exist. Run 'dvc repro prepare' first, then rerun with -SeedData."
    }

    Write-Step "Seeding Kubeflow data PVC from data/processed"
    $podName = "vehicle-data-seeder"

    & $Kubectl delete pod $podName -n kubeflow --ignore-not-found=true | Out-Null

    $podYaml = @"
apiVersion: v1
kind: Pod
metadata:
  name: $podName
  namespace: kubeflow
spec:
  restartPolicy: Never
  containers:
  - name: seeder
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: vehicle-data
"@

    try {
        $podYaml | & $Kubectl apply -f -
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create data seeder pod."
        }

        Invoke-Checked $Kubectl "wait" "--for=condition=ready" "pod/$podName" "-n" "kubeflow" "--timeout=60s"
        Invoke-Checked $Kubectl "exec" "-n" "kubeflow" $podName "--" "sh" "-c" "rm -rf /data/processed"
        Invoke-Checked $Kubectl "cp" "data/processed" "${podName}:/data/processed" "-n" "kubeflow"
    }
    finally {
        & $Kubectl delete pod $podName -n kubeflow --ignore-not-found=true | Out-Null
    }
}

$minikubeKnownPaths = @(
    "C:\Program Files\Kubernetes\Minikube\minikube.exe",
    "C:\ProgramData\chocolatey\bin\minikube.exe"
)

$MinikubeCmd = Resolve-Tool "minikube" -KnownPaths $minikubeKnownPaths
if (-not $MinikubeCmd -and $InstallMinikube) {
    Write-Step "Installing Minikube with winget"
    $WingetCmd = Resolve-Tool "winget"
    if (-not $WingetCmd) {
        throw "winget was not found. Install Minikube manually, then rerun this script."
    }

    Invoke-Checked $WingetCmd "install" "--id" "Kubernetes.minikube" "-e"
    $MinikubeCmd = Resolve-Tool "minikube" -KnownPaths $minikubeKnownPaths
}

if (-not $MinikubeCmd) {
    throw "Minikube was not found. Install it or rerun this script with -InstallMinikube."
}

$KubectlCmd = Resolve-Tool "kubectl"
if (-not $KubectlCmd) {
    throw "kubectl was not found. Install kubectl, then rerun this script."
}

$DockerCmd = Resolve-Tool "docker"
if (($Driver -eq "docker" -or $BuildImages) -and -not $DockerCmd) {
    throw "Docker was not found. Docker is required for the docker Minikube driver and image builds."
}

Write-Step "Using tools"
Write-Host "Repo:      $RepoRoot"
Write-Host "minikube:  $MinikubeCmd"
Write-Host "kubectl:   $KubectlCmd"
if ($DockerCmd) {
    Write-Host "docker:    $DockerCmd"
}

if (-not $SkipSdkInstall) {
    Write-Step "Installing local Kubeflow Pipelines Python SDK"
    $PythonCmd = Resolve-ProjectPython
    Invoke-Checked $PythonCmd "-m" "pip" "install" "kfp>=2.0" "kfp-kubernetes"
}

Start-MinikubeIfNeeded -Minikube $MinikubeCmd
Install-KubeflowPipelines -Kubectl $KubectlCmd
Apply-ProjectResources -Kubectl $KubectlCmd
Build-And-LoadImages -Docker $DockerCmd -Minikube $MinikubeCmd
Seed-DataPvc -Kubectl $KubectlCmd

Write-Step "Done"
Write-Host "Kubeflow Pipelines UI:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
Write-Host "  http://localhost:8080"
Write-Host ""
Write-Host "After port-forwarding the UI, submit the pipeline with:"
Write-Host "  .\.venv\Scripts\python.exe pipeline\pipeline.py"

if ($PortForward) {
    Write-Step "Starting Kubeflow Pipelines UI port-forward"
    Write-Host "Open http://localhost:8080. Leave this terminal running."
    Invoke-Checked $KubectlCmd "port-forward" "-n" "kubeflow" "svc/ml-pipeline-ui" "8080:80"
}
