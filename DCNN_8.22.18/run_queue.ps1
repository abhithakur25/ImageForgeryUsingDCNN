<#
  run_queue.ps1  -  Execution queue for the transfer-learning backbone sweep.

  Runs Train_Transfer.m once per backbone, sequentially, on the GPU.
  - Resumable: a job whose Results\Transfer_<Tag>\SUMMARY.txt already exists is
    skipped (use -Force to re-run everything).
  - Shows a live progress bar for the whole queue AND for the current job's
    epochs, both on the console (Write-Progress + text) and in
    Results\queue_status.txt so progress can be read at any time.

  Usage:
    powershell -ExecutionPolicy Bypass -File run_queue.ps1
    powershell -ExecutionPolicy Bypass -File run_queue.ps1 -Force
#>
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$root    = $PSScriptRoot
$matlab  = 'F:\Program Files\MATLAB\R2022b\bin\matlab.exe'
$resRoot = Join-Path $root 'Results'
$statusF = Join-Path $resRoot 'queue_status.txt'

# ---- The queue (same 30-epoch / LR 1e-4 protocol as the resnet18 run) --------
$queue = @(
    @{ Net='resnet50'; Epochs=30; LR='1e-4'; MBS=64; Tag='resnet50' }
    @{ Net='alexnet';  Epochs=30; LR='1e-4'; MBS=64; Tag='alexnet'  }
    @{ Net='vgg16';    Epochs=30; LR='1e-4'; MBS=24; Tag='vgg16'    }  # smaller batch: 138M params on 8GB GPU
)

function Get-Bar([double]$frac, [int]$width = 40) {
    if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
    $f = [int][math]::Round($frac * $width)
    ('#' * $f) + ('-' * ($width - $f))
}

function Get-CurrentEpoch([string]$logFile, [int]$maxEpochs) {
    if (-not (Test-Path $logFile)) { return 0 }
    # Training-progress rows look like: |   12 |   1632 | 00:09:30 | ...
    $last = Select-String -Path $logFile -Pattern '^\|\s*(\d+)\s*\|\s*\d+\s*\|' -ErrorAction SilentlyContinue |
            Select-Object -Last 1
    if ($last) { return [int]$last.Matches[0].Groups[1].Value }
    return 0
}

function Write-Status([int]$jobIdx, [int]$total, [hashtable]$job, [double]$jobFrac, [string]$phase, [datetime]$startedAll) {
    $overall = ($jobIdx + $jobFrac) / $total
    $elapsed = (Get-Date) - $startedAll
    $lines = @()
    $lines += "DCNN transfer-learning execution queue"
    $lines += ("updated : {0}   elapsed {1:hh\:mm\:ss}" -f (Get-Date -Format 'HH:mm:ss'), $elapsed)
    $lines += ""
    $lines += ("OVERALL  [{0}] {1,5:P0}   job {2}/{3}" -f (Get-Bar $overall), $overall, ($jobIdx + 1), $total)
    $lines += ("CURRENT  [{0}] {1,5:P0}   {2}  ({3})" -f (Get-Bar $jobFrac), $jobFrac, $job.Tag, $phase)
    $lines += ""
    $lines += "queue:"
    for ($k = 0; $k -lt $total; $k++) {
        $mark = if ($k -lt $jobIdx) { '[x]' } elseif ($k -eq $jobIdx) { '[>]' } else { '[ ]' }
        $lines += ("  {0} {1}" -f $mark, $queue[$k].Tag)
    }
    $text = $lines -join "`r`n"
    Set-Content -Path $statusF -Value $text -Encoding utf8
    Write-Progress -Id 0 -Activity "Execution queue" -Status ("job {0}/{1}: {2}" -f ($jobIdx+1), $total, $job.Tag) -PercentComplete ([int]($overall*100))
    Write-Progress -Id 1 -ParentId 0 -Activity ("Training {0}" -f $job.Tag) -Status $phase -PercentComplete ([int]($jobFrac*100))
}

$total     = $queue.Count
$startedAll = Get-Date
Write-Host "=== Execution queue: $total job(s) ===" -ForegroundColor Cyan

for ($i = 0; $i -lt $total; $i++) {
    $job    = $queue[$i]
    $resDir = Join-Path $resRoot ("Transfer_" + $job.Tag)
    $summary = Join-Path $resDir 'SUMMARY.txt'
    $logFile = Join-Path $resDir 'run_log.txt'

    if ((Test-Path $summary) -and -not $Force) {
        Write-Host ("[{0}/{1}] {2}  -- already done, skipping" -f ($i+1), $total, $job.Tag) -ForegroundColor DarkGray
        Write-Status $i $total $job 1.0 'done (cached)' $startedAll
        continue
    }

    Write-Host ("[{0}/{1}] {2}  -- starting ({3} epochs, LR {4}, batch {5})" -f `
        ($i+1), $total, $job.Tag, $job.Epochs, $job.LR, $job.MBS) -ForegroundColor Green
    Write-Status $i $total $job 0.0 'starting MATLAB...' $startedAll

    $cmd = "BASENET='$($job.Net)'; EPOCHS=$($job.Epochs); LR0=$($job.LR); MBS=$($job.MBS); TAG='$($job.Tag)'; Train_Transfer"
    $arg = '-batch "' + $cmd + '"'   # single arg string; array form mangles ; ' = chars
    $proc = Start-Process -FilePath $matlab -ArgumentList $arg `
                          -WorkingDirectory $root -PassThru -WindowStyle Hidden

    $frac = 0
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 10
        $ep   = Get-CurrentEpoch $logFile $job.Epochs
        $frac = if ($job.Epochs -gt 0) { $ep / $job.Epochs } else { 0 }
        Write-Status $i $total $job $frac ("epoch {0}/{1}" -f $ep, $job.Epochs) $startedAll
    }

    $ok = (Test-Path $logFile) -and (Select-String -Path $logFile -Pattern 'DONE_TRAIN_OPTIMIZED' -Quiet)
    if ($ok) {
        Write-Host ("[{0}/{1}] {2}  -- DONE" -f ($i+1), $total, $job.Tag) -ForegroundColor Green
        Write-Status $i $total $job 1.0 'done' $startedAll
    } else {
        Write-Host ("[{0}/{1}] {2}  -- FAILED (exit {3}); see run_log.txt" -f `
            ($i+1), $total, $job.Tag, $proc.ExitCode) -ForegroundColor Red
        Write-Status $i $total $job $frac ("FAILED exit $($proc.ExitCode)") $startedAll
    }
}

Write-Progress -Id 1 -Activity 'done' -Completed
Write-Progress -Id 0 -Activity 'done' -Completed
$elapsed = (Get-Date) - $startedAll
Add-Content -Path $statusF -Value ("`r`n=== queue finished in {0:hh\:mm\:ss} ===" -f $elapsed) -Encoding utf8
Write-Host ("=== Queue finished in {0:hh\:mm\:ss} ===" -f $elapsed) -ForegroundColor Cyan

# Aggregate everything (incl. the new backbones) into COMPARISON.* / FINAL_REPORT
Write-Host "Running Compare_Models to refresh COMPARISON..." -ForegroundColor Cyan
& $matlab -batch "Compare_Models" 2>&1 | Out-Null
Write-Host "All done." -ForegroundColor Cyan
