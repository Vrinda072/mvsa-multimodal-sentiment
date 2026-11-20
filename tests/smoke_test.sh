#!/usr/bin/env bash
# Runs the whole pipeline on small synthetic data with one seed and a tiny CLIP model.
# This checks that everything runs, in a few minutes. The scores it produces mean nothing.
set -euo pipefail

# Work in a temporary folder that is deleted when the test ends, whatever happens
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Create synthetic posts, then run every notebook on them with one seed and a tiny random CLIP model
python "$REPO/tests/make_synthetic_data.py" "$WORK/data/MVSA_Single"
MVSA_ROOT="$WORK/data" MVSA_RUNS="$WORK/runs" MVSA_QUICK=1 MVSA_INSTALL_EXTRAS=0 \
  MVSA_CLIP_MODEL=hf-internal-testing/tiny-random-CLIPModel \
  bash "$REPO/scripts/run_all.sh"

# Check the key outputs exist and are not empty
for file in preprocessed/posts_preprocessed.csv models/all_runs.csv models/chosen_model.json models/bias_variance_curve.png \
            models/learning_curve.csv models/bootstrap_intervals.csv models/calibration.csv models/risk_coverage.csv; do
  test -s "$WORK/runs/$file" || { echo "Missing output: $file"; exit 1; }
done
echo "Smoke test passed"
