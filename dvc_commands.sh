#!/usr/bin/env bash
# Helper script to (initially) track and push datasets with DVC.
# Idempotent: safe to re-run in CI (GitHub Actions) or locally.
# Customize the paths you want DVC to manage below.
set -euo pipefail

# 1. Initialize DVC repository (only first time)
if [ ! -d .dvc ]; then
  dvc init -q
fi

# 2. (Optional) pull existing cache before adding (improves deduplication)
(dvc pull >/dev/null 2>&1 || true)

# 3. Track raw / processed datasets (adjust as needed)
# Add or remove lines for other directories (large, versioned data). Each 'dvc add'
# creates a .dvc metafile that is committed to git while actual data lives in the DVC cache / remote.
if [ -d resources/bacteries_2024/tsv ]; then
  dvc add resources/bacteries_2024/tsv
fi
if [ -d results/multi ]; then
  dvc add results/multi
fi

# 4. Commit DVC metafiles (no-op if nothing changed)
# (We do not commit here in CI if we prefer a separate step; locally it's convenient.)
if git diff --name-only --exit-code >/dev/null 2>&1; then
  echo "No dataset changes detected for DVC."
else
  git add resources/bacteries_2024/tsv.dvc results/multi.dvc .gitignore 2>/dev/null || true
fi

# 5. Configure remote (first time). Expect environment variable DVC_REMOTE_URL (e.g. s3://bucket/path or ssh://host/path)
# In GitHub Actions define a secret DVC_REMOTE_URL and provider credentials (AWS, GCP, etc.).
if ! dvc remote list | grep -q '^storage'; then
  if [ "${DVC_REMOTE_URL:-}" != "" ]; then
    dvc remote add -d storage "${DVC_REMOTE_URL}" || true
  else
    echo "Warning: DVC_REMOTE_URL not set; skipping remote add (data will remain local)."
  fi
fi

# 6. Push data to remote (if configured)
if dvc remote list | grep -q '^storage'; then
  dvc push
else
  echo "No remote named 'storage' configured; skipped dvc push."
fi

echo "DVC dataset tracking script finished."
