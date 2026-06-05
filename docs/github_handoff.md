# GitHub Handoff

This repository is prepared so the source tree can be pushed to a public GitHub repository without large generated artifacts.

## Before Push

Fill report metadata:

```bash
python scripts/report/fill_report_metadata.py --metadata report/metadata.json
bash scripts/report/build_report.sh
```

Use `report/metadata.example.json` as the template. The real `report/metadata.json` is ignored by Git.

## Push

```bash
git remote add origin https://github.com/<account>/<repo>.git
git branch -M main
git push -u origin main
```

## Large Artifacts

Do not push these directories to GitHub:

```text
data/
external/
results/
weights/
tools/
submission/
.venvs/
```

Upload `weights/cv_hw3_task2_act_weights.tar.gz` to a cloud drive and put that URL into `report/metadata.json` before rebuilding the report.

## Final Gate

Run:

```bash
python scripts/report/check_submission_ready.py --json results/submission_readiness.json
```

The gate is expected to fail until real phone-captured Task 1 inputs and report metadata are provided.
