# Results

All numbers come from the full five-seed Kaggle run. The data behind each figure is in `results/metrics/`, and `python scripts/make_figures.py` redraws the figures.

## Dataset

![Counts of labels, agreement groups and data splits](../results/figures/01_dataset_overview.png)

The classes are imbalanced and text and image labels often disagree, on 46.8% of posts.

![One feature before and after log scaling and standardisation](../results/figures/02_feature_scaling.png)

## Model comparison

![Test macro-F1 for every classifier and fusion setup](../results/figures/03_model_comparison.png)

Logistic regression with late fusion had the highest validation macro-F1 (0.619) and was selected. It scores 0.603 macro-F1 and 0.696 accuracy on the test set, compared with 0.504 and 0.622 for the baseline.

The MLP with early + meta scored highest on test (0.655), but its validation score was 0.613, so it was not selected. Choosing it on test score would invalidate the test set as an unbiased estimate.

## Per-class results

![Precision, recall and F1 per class for the chosen model](../results/figures/04_per_class_metrics.png)

![Confusion matrix for the chosen model](../results/figures/05_confusion_matrix.png)

Neutral is the weak class. The model found 19 of the 39 neutral test posts, but only 19 of its 69 neutral predictions were correct.

## Did gated fusion help?

No. For logistic regression, gated fusion was less accurate than early fusion on both agreeing posts (0.755 vs 0.779) and soft-mismatch posts (0.569 vs 0.639). Compared with early fusion, it lowered macro-precision in all four classifier families and raised macro-recall in three.

| Family | Early only correct | Gated only correct | McNemar p-value |
|---|---|---|---|
| Logistic regression | 38 | 19 | 0.016 |
| Linear SVM | 57 | 31 | 0.007 |
| Random forest | 78 | 60 | 0.148 |
| MLP | 45 | 27 | 0.044 |

![Coefficients of the gate for each input and class](../results/figures/07_gate_coefficients.png)

The gate relies on the two sets of class probabilities and gives the Jensen-Shannon divergence almost no weight, so it behaves much like an average of the two modalities. These inputs are correlated and the coefficients come from one seed, so this is supporting evidence rather than proof.

## Missing modalities

![Macro-F1 with both modalities, the image masked and the text masked](../results/figures/06_missing_modality.png)

Masking the image hurt macro-F1 in all three fusion setups, while masking the text changed it far less. The mean embedding isn't a neutral input, because it pushes predictions towards the majority class. This test therefore doesn't show how the model would behave with truly missing data.

## Bias-variance trade-off and regularisation

![Training and validation macro-F1 across regularisation strengths](../results/figures/08_bias_variance_curve.png)

Stronger L2 regularisation narrowed early fusion's train-validation gap from 0.46 to 0.10, showing that its main risk is high variance. At large `C`, it almost memorises the training set (0.991) while validation stays near 0.53.

The chosen `C = 0.01` is the strongest value in the tuning grid. Late fusion reaches 0.627 at `C = 0.001`, compared with 0.619 at the selected value. The reported model keeps the pre-specified grid.

## Limitations

The top validation scores differ by only 0.006 on 446 posts, and no confidence intervals were computed. The McNemar tests, gate coefficients, confusion matrix and robustness results each use a single seed.

## Reproducibility

The six notebooks were rerun in order on the full dataset and compared with the results above. The preprocessing counts, the chosen model and its scores, and every logistic regression, linear SVM and random forest score matched exactly.

Only the MLP differed. Its tuned late-fusion weight was 0.3 rather than 0.5, which moved MLP late fusion from 0.632 to 0.618 test macro-F1. Its other macro-F1 scores moved by 0.002 or less, and MLP early + meta did not change at 3 decimal places. The bias-variance curve moved by at most 0.003, at `C` of 10 or more.

The code was the same in both runs, but the model training ran on a CPU machine the first time and a GPU machine the second. That is the likely cause, because MLP training can be sensitive to the hardware, though this wasn't tested separately.

## Where to find each result

| Result | Notebook in `notebooks/02_modeling/` |
|---|---|
| Model comparison, per-class and per-group results | `01_training_and_model_selection.ipynb` |
| McNemar tests, missing modalities, gate coefficients, confusion matrix | `02_significance_and_robustness.ipynb` |
| Bias-variance curve and tuned hyperparameters | `03_bias_variance_and_regularisation.ipynb` |
