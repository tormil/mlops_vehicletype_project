<#
.SYNOPSIS
Installs Prometheus/Grafana and applies the vehicle classifier PodMonitor.
#>

[CmdletBinding()]
param(
    [string]$Namespace = "monitoring",
    [string]$ReleaseName = "prom",
    [string]$HelmPath,
    [switch]$SkipPodMonitor,
    [switch]$PortForward
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

function Resolve-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$ExplicitPath,
        [string[]]$KnownPaths = @()
    )

    if ($ExplicitPath) {
        if (Test-Path $ExplicitPath) {
            return (Resolve-Path $ExplicitPath).Path
        }

        throw "$Name was not found at: $ExplicitPath"
    }

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

function Find-Helm {
    $knownPaths = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\helm.exe",
        "$env:USERPROFILE\scoop\shims\helm.exe",
        "C:\ProgramData\chocolatey\bin\helm.exe",
        "C:\Program Files\Helm\helm.exe"
    )

    $helm = Resolve-Tool "helm" -ExplicitPath $HelmPath -KnownPaths $knownPaths
    if ($helm) {
        return $helm
    }

    $wingetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetPackages) {
        $installedHelm = Get-ChildItem -Path $wingetPackages -Filter "helm.exe" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
        if ($installedHelm) {
            return $installedHelm
        }
    }

    throw "helm was not found. Install it, reopen PowerShell, or rerun with -HelmPath C:\path\to\helm.exe."
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

$Helm = Find-Helm
$Kubectl = Resolve-Tool "kubectl"
if (-not $Kubectl) {
    throw "kubectl was not found. Install kubectl and rerun this script."
}

Write-Host "==> Installing Prometheus/Grafana" -ForegroundColor Cyan
Write-Host "Using Helm: $Helm"

$existingNamespace = & $Kubectl get namespace $Namespace --ignore-not-found=true -o name
if (-not $existingNamespace) {
    Invoke-Checked $Kubectl "create" "namespace" $Namespace
}

Invoke-Checked $Helm "repo" "add" "prometheus-community" "https://prometheus-community.github.io/helm-charts" "--force-update"
Invoke-Checked $Helm "repo" "update"
Invoke-Checked $Helm "upgrade" "--install" $ReleaseName "prometheus-community/kube-prometheus-stack" "-n" $Namespace `
    "--set" "prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false" `
    "--set" "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false"

Invoke-Checked $Kubectl "wait" "--for=condition=established" "--timeout=120s" "crd/podmonitors.monitoring.coreos.com"
Invoke-Checked $Kubectl "rollout" "status" "deployment/$ReleaseName-grafana" "-n" $Namespace "--timeout=300s"

if (-not $SkipPodMonitor) {
    Invoke-Checked $Kubectl "apply" "-f" "deployment/k8s/vehicle-classifier-podmonitor.yaml"
}

Write-Host ""
Write-Host "Grafana:" -ForegroundColor Cyan
Write-Host "  username: admin"
Write-Host "  password:"
$passwordCommand = 'kubectl get secret -n ' + $Namespace + ' ' + $ReleaseName + '-grafana -o jsonpath="{.data.admin-password}" | % { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }'
Write-Host "    $passwordCommand"
Write-Host "  kubectl port-forward -n $Namespace svc/$ReleaseName-grafana 3000:80"
Write-Host "  http://localhost:3000"

if ($PortForward) {
    Write-Host ""
    Write-Host "Starting Grafana port-forward. Leave this terminal running." -ForegroundColor Cyan
    Invoke-Checked $Kubectl "port-forward" "-n" $Namespace "svc/$ReleaseName-grafana" "3000:80"
}
