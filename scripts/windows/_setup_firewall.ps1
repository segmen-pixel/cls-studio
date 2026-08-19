# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
<#
.SYNOPSIS
    Configure Windows Firewall so other PCs on a Private-profile LAN can
    reach the cls-studio API + UI (single port 8791).

.DESCRIPTION
    Called from start_local_windows.bat on first LAN startup (must be
    invoked elevated; the bat self-elevates via Start-Process -Verb RunAs).

    - Disables any pre-existing Inbound Block rule that targets the venv's
      base python.exe (otherwise Block wins over our Allow rules).
    - Adds an idempotent Inbound Allow rule tagged "cls-studio" for
      TCP 8791 scoped to the Private profile.

.PARAMETER BasePython
    Absolute path to the python.exe the venv launchers re-exec to (read
    from .venv-*/pyvenv.cfg). Pass an empty string to skip the Block-rule
    cleanup step.
#>
param(
    [string]$BasePython = ""
)

$ErrorActionPreference = "Stop"

if ($BasePython) {
    $blocks = Get-NetFirewallRule -Direction Inbound -Action Block -ErrorAction SilentlyContinue |
        Where-Object {
            $_ | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue |
                Where-Object { $_.Program -ieq $BasePython }
        }
    foreach ($r in $blocks) {
        Disable-NetFirewallRule -Name $r.Name
        Write-Host ("disabled Block rule: " + $r.DisplayName)
    }
    if (-not $blocks) {
        Write-Host "no conflicting Block rules found for $BasePython"
    }
}

$specs = @(
    @{ Name = 'cls-studio LAN api 8791'; Port = 8791 }
)
foreach ($s in $specs) {
    $existing = Get-NetFirewallRule -DisplayName $s.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host ("Allow rule already present: " + $s.Name)
        continue
    }
    New-NetFirewallRule `
        -DisplayName $s.Name `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $s.Port `
        -Profile Private `
        -Group 'cls-studio' | Out-Null
    Write-Host ("added Allow rule: " + $s.Name)
}
