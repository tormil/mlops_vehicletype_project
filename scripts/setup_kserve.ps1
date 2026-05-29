<#
.SYNOPSIS
Installs KServe and its local Minikube dependencies.
#>

[CmdletBinding()]
param(
    [string]$KnativeVersion = "1.20.0",
    [string]$CertManagerVersion = "1.16.1",
    [string]$KServeVersion = "0.17.0",
    [switch]$SkipDependencies,
    [switch]$SkipClusterResources,
    [switch]$DeployInferenceService
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

function Resolve-Tool {
    param([Parameter(Mandatory = $true)][string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name was not found. Install $Name and rerun this script."
    }

    return $cmd.Source
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
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

function Invoke-CheckedRetry {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [int]$Attempts = 5,
        [int]$DelaySeconds = 15
    )

    for ($i = 1; $i -le $Attempts; $i++) {
        Write-Host "> $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkGray
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -eq 0) {
            return
        }

        if ($i -eq $Attempts) {
            throw "Command failed after $Attempts attempts: $FilePath $($ArgumentList -join ' ')"
        }

        Write-Host "Command failed; retrying in $DelaySeconds seconds..."
        Start-Sleep -Seconds $DelaySeconds
    }
}

function Ensure-Namespace {
    param([Parameter(Mandatory = $true)][string]$Name)

    $existing = & $Kubectl get namespace $Name --ignore-not-found=true -o name
    if ($LASTEXITCODE -ne 0) {
        throw "Could not check namespace: $Name"
    }

    if (-not $existing) {
        Invoke-Checked $Kubectl "create" "namespace" $Name
    }
}

function Wait-NamespacePods {
    param(
        [Parameter(Mandatory = $true)][string]$Namespace,
        [int]$TimeoutSeconds = 300
    )

    Invoke-Checked $Kubectl "wait" "--for=condition=ready" "pod" "--all" "-n" $Namespace "--timeout=${TimeoutSeconds}s"
}

function Wait-Deployment {
    param(
        [Parameter(Mandatory = $true)][string]$Namespace,
        [Parameter(Mandatory = $true)][string]$Name,
        [int]$TimeoutSeconds = 300
    )

    Invoke-Checked $Kubectl "rollout" "status" "deployment/$Name" "-n" $Namespace "--timeout=${TimeoutSeconds}s"
}

function Wait-EndpointSlice {
    param(
        [Parameter(Mandatory = $true)][string]$Namespace,
        [Parameter(Mandatory = $true)][string]$ServiceName,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $addresses = & $Kubectl get endpointslice -n $Namespace `
            -l "kubernetes.io/service-name=$ServiceName" `
            -o jsonpath="{.items[*].endpoints[*].addresses[*]}" 2>$null
        if ($LASTEXITCODE -eq 0 -and $addresses) {
            return
        }
        Start-Sleep -Seconds 3
    }

    throw "Timed out waiting for EndpointSlice addresses for $Namespace/$ServiceName."
}

$Kubectl = Resolve-Tool "kubectl"

if (-not $SkipDependencies) {
    Write-Step "Installing Istio for Knative"
    $netIstioBase = "https://github.com/knative/net-istio/releases/download/knative-v$KnativeVersion"
    Invoke-Checked $Kubectl "apply" "-l" "knative.dev/crd-install=true" "-f" "$netIstioBase/istio.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "$netIstioBase/istio.yaml"
    Wait-NamespacePods -Namespace "istio-system"

    Write-Step "Installing Knative Serving"
    $servingBase = "https://github.com/knative/serving/releases/download/knative-v$KnativeVersion"
    Invoke-Checked $Kubectl "apply" "-f" "$servingBase/serving-crds.yaml"
    Invoke-Checked $Kubectl "apply" "-f" "$servingBase/serving-core.yaml"
    Wait-NamespacePods -Namespace "knative-serving"

    Write-Step "Installing Knative net-istio"
    Invoke-Checked $Kubectl "apply" "-f" "$netIstioBase/net-istio.yaml"
    Wait-NamespacePods -Namespace "knative-serving"

    Write-Step "Installing cert-manager"
    Invoke-Checked $Kubectl "apply" "-f" "https://github.com/cert-manager/cert-manager/releases/download/v$CertManagerVersion/cert-manager.yaml"
    Wait-NamespacePods -Namespace "cert-manager"
}
else {
    Write-Host "Skipping Istio, Knative, and cert-manager because -SkipDependencies was supplied."
}

Write-Step "Installing KServe $KServeVersion"
Ensure-Namespace -Name "kserve"
$kserveBase = "https://github.com/kserve/kserve/releases/download/v$KServeVersion"
Invoke-Checked $Kubectl "apply" "--server-side" "--force-conflicts" "-f" "$kserveBase/kserve.yaml"

Write-Step "Waiting for KServe webhooks"
Wait-Deployment -Namespace "kserve" -Name "kserve-controller-manager"
Wait-Deployment -Namespace "kserve" -Name "llmisvc-controller-manager"
Wait-EndpointSlice -Namespace "kserve" -ServiceName "kserve-webhook-server-service"
Wait-EndpointSlice -Namespace "kserve" -ServiceName "llmisvc-webhook-server-service"
Start-Sleep -Seconds 15

if (-not $SkipClusterResources) {
    Write-Step "Installing KServe cluster resources"
    Invoke-CheckedRetry -FilePath $Kubectl -ArgumentList @(
        "apply", "--server-side", "--force-conflicts",
        "-f", "$kserveBase/kserve-cluster-resources.yaml"
    )
}
else {
    Write-Host "Skipping KServe cluster resources because -SkipClusterResources was supplied."
}

Write-Step "Allowing local dev image tags in Knative"
$localImagePatch = '{"data":{"registries-skipping-tag-resolving":"kind.local,ko.local,dev.local,docker.io,index.docker.io"}}'
$patchFile = Join-Path ([System.IO.Path]::GetTempPath()) "knative-config-deployment-patch.json"
[System.IO.File]::WriteAllText($patchFile, $localImagePatch, [System.Text.UTF8Encoding]::new($false))
try {
    Invoke-Checked $Kubectl "patch" "configmap" "config-deployment" "-n" "knative-serving" "--type=merge" `
        "--patch-file" $patchFile
}
finally {
    Remove-Item -LiteralPath $patchFile -ErrorAction SilentlyContinue
}

if ($DeployInferenceService) {
    Write-Step "Deploying vehicle classifier InferenceService"
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/inferenceservice.yaml"
}

Write-Step "Done"
Write-Host "Verify KServe:"
Write-Host "  kubectl get crd inferenceservices.serving.kserve.io"
Write-Host "  kubectl get pods -n kserve"
Write-Host ""
Write-Host "Deploy the vehicle classifier when a champion model exists:"
Write-Host "  kubectl apply -f deployment/k8s/inferenceservice.yaml"
Write-Host "  kubectl get inferenceservice -n vehicle -w"
