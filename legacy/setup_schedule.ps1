<#
  Registers a Windows scheduled task that runs the weekly pull every Tuesday
  at 12:00 local time, and again Wednesday at 08:00 to backfill snap counts
  that Pro Football Reference posts late.

  Run this ONCE:   .\setup_schedule.ps1
  To remove:       .\setup_schedule.ps1 -Remove
#>
param([switch]$Remove)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$names = @("Fantasy weekly pull (Tue)", "Fantasy snap backfill (Wed)", "Fantasy status refresh (Fri)")

if ($Remove) {
    foreach ($n in $names) {
        if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $n -Confirm:$false
            Write-Host "Removed: $n"
        }
    }
    return
}

$python = (Get-Command python -ErrorAction Stop).Source
Write-Host "Using python: $python"
Write-Host "Project root: $root"

$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m ff.run_weekly" -WorkingDirectory $root
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Friday afternoon is when final injury designations and the last practice
# report land — before that, availability data is incomplete.
$triggers = @{
    "Fantasy weekly pull (Tue)"     = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday   -At 12:00
    "Fantasy snap backfill (Wed)"   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday -At 08:00
    "Fantasy status refresh (Fri)"  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday    -At 16:00
}

foreach ($n in $names) {
    if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false
    }
    Register-ScheduledTask -TaskName $n -Action $action -Trigger $triggers[$n] `
        -Settings $settings -Description "Fantasy volume tracker" | Out-Null
    Write-Host "Registered: $n"
}

Write-Host ""
Write-Host "Done. Check them in Task Scheduler, or run now with:"
Write-Host "  Start-ScheduledTask -TaskName 'Fantasy weekly pull (Tue)'"
