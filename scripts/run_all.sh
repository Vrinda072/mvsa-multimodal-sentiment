#!/usr/bin/env bash
# Runs all six notebooks in order. The committed notebooks are not changed: executed copies and
# all generated files go to a separate folder (runs/ locally, /kaggle/working on Kaggle).
#
# Usage: MVSA_ROOT=/path/to/MVSA_Single bash scripts/run_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d /kaggle/working ]; then
  RUNS="${MVSA_RUNS:-/kaggle/working}"
  export MVSA_ROOT="${MVSA_ROOT:-/kaggle/input}"
else
  RUNS="${MVSA_RUNS:-$REPO/runs}"
  export MVSA_ROOT="${MVSA_ROOT:-$REPO/data/raw}"
fi
export MVSA_OUT="$RUNS/preprocessed"
export MVSA_PRE="$RUNS/preprocessed"
export MVSA_MODEL_OUT="$RUNS/models"
export MVSA_INTERIM="$RUNS/interim"

for notebook in \
  01_preprocessing/01_data_cleaning_and_labels \
  01_preprocessing/02_train_val_test_split \
  01_preprocessing/03_feature_engineering \
  02_modeling/01_training_and_model_selection \
  02_modeling/02_significance_and_robustness \
  02_modeling/03_bias_variance_and_regularisation
do
  echo "Running $notebook"
  python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir "$RUNS/executed/$(dirname "$notebook")" "$REPO/notebooks/$notebook.ipynb"
done
echo "Done. Results are in $RUNS"
