# Legacy PowerShell (retired)

These ran the pipeline on James's Windows PC before the migration to GitHub
Actions. They are kept for reference and for one remaining job:

- `setup_schedule.ps1` — registered the Windows scheduled tasks. Run it once
  with `-Remove` to retire them, so the workflow is the only writer:
  `cd $env:USERPROFILE\Projects\fantasy ; .\setup_schedule.ps1 -Remove`
- `run_weekly.ps1` — local wrapper around `python -m ff.run_weekly`.

Nothing in this folder is used by the workflow. The schedule now lives in
`.github/workflows/pipeline.yml`.
