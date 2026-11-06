# Multimodal Sentiment Analysis on MVSA-Single

BITS F464 Machine Learning project.

The project classifies Twitter text-image posts as negative, neutral or positive. It compares four classifiers and several ways of combining text and image, all on frozen CLIP ViT-B/32 embeddings.

| File | What it does |
|---|---|
| `notebooks/mvsa_preprocessing.ipynb` | Cleans the data, derives labels, splits it 8:1:1 and builds the features |
| `notebooks/mvsa_model_design.ipynb` | Trains, tunes and evaluates the models, using the preprocessing output |
| `results/bias_variance_curve.png` | Bias-variance figure from the full run |
| `requirements.txt` | Python dependencies |

Both notebooks include the saved outputs from their full Kaggle runs, so you can read the results without rerunning them.

## Dataset

The dataset is not included in this repository because of its size. Download MVSA-Single from Kaggle: [vincemarcs/mvsasingle](https://www.kaggle.com/datasets/vincemarcs/mvsasingle). It should contain `labelResultAll.txt` and a `data/` folder with one `.txt` and one `.jpg` file per post.

## How to run

Run the preprocessing notebook first, because the model notebook uses its output.

### On Kaggle (recommended)

These are the settings used for the results below.

1. **Preprocessing**
   1. Create a notebook and choose **File > Import Notebook**, then select `mvsa_preprocessing.ipynb`.
   2. Choose **Add Input** and add `vincemarcs/mvsasingle`.
   3. In **Settings**, set **Accelerator** to GPU and turn **Internet** on.
   4. Choose **Save Version > Save & Run All**. This took about 4 minutes on a T4 GPU.
2. **Save the output as a dataset:** open the saved version's **Output** tab and choose **New Dataset**.
3. **Models**
   1. Import `mvsa_model_design.ipynb` into a new notebook.
   2. Choose **Add Input** and add the dataset from step 2. The notebook finds the `preprocessed/` folder by itself.
   3. Choose **Save Version > Save & Run All**. No GPU is needed. All five seeds took about 30 minutes on CPU.

A warning about a missing `HF_TOKEN` is harmless. CLIP still downloads.

For a faster test with one seed, add this as the first cell of the model notebook:

```python
import os
os.environ["MVSA_QUICK"] = "1"
```

### On your own computer

Python 3.10 or newer is needed. Run these commands from the repository folder, replacing the dataset path with your own:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MVSA_ROOT=/path/to/MVSA_Single
export MVSA_OUT="$PWD/preprocessed"
export MVSA_PRE="$PWD/preprocessed"
export MVSA_MODEL_OUT="$PWD/models"

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/mvsa_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/mvsa_model_design.ipynb
```

The preprocessing notebook needs internet access to download CLIP. Without a GPU, CLIP embedding can take 20 minutes or more.

## Dependencies

Python packages are listed in `requirements.txt`. The results below came from Kaggle's "Latest Container Image" environment, so versions are not pinned. `emoji`, `wordsegment`, `ftfy` and `nltk` are optional: without them the preprocessing notebook uses simpler fallbacks or skips VADER scores, which can change results slightly.

## Expected outputs

### Preprocessing

The notebook writes a `preprocessed/` folder:

- `posts_preprocessed.csv` and `preprocessing_report.json`
- `features/`: IDs, CLIP embeddings, TF-IDF, colour histograms and engineered features for each split
- `artifacts/`: the fitted TF-IDF vectoriser and scaler
- `figures/`: a plot of the feature scaling

| Check | Expected value |
|---|---|
| Posts after removing duplicates | 4,839 of 4,869 |
| Agreement / soft mismatch / hard mismatch | 2,576 / 1,908 / 355 |
| Train / validation / test | 3,591 / 446 / 447, with the 355 hard-mismatch posts held out |

### Models

The notebook prints its results and writes a `models/` folder:

| File | Contents |
|---|---|
| `all_runs.csv`, `summary_mean_over_seeds.csv` | Every metric for each model, per seed and averaged |
| `per_class_metrics.csv` | Precision, recall and F1 per class for the chosen model |
| `tuned_hyperparameters.json`, `chosen_model.json` | Tuned settings and the selected model |
| `robustness_missing_modality.csv` | Scores with the text or image masked |
| `gate_coefficients.csv` | What the gated fusion model learned |
| `bias_variance_curve.csv`, `bias_variance_curve.png` | Training and validation macro-F1 across regularisation strengths |

The model is chosen by validation macro-F1, not test score. The full run selects **logistic regression with late fusion**:

| Metric | Value |
|---|---|
| Validation macro-F1 (used for selection) | 0.619 |
| Test macro-F1 | 0.603 |
| Test accuracy | 0.696 |
| Macro precision / macro recall | 0.591 / 0.637 |

| Class | Precision | Recall | F1 | Test posts |
|---|---|---|---|---|
| Negative | 0.653 | 0.701 | 0.676 | 134 |
| Neutral | 0.275 | 0.487 | 0.352 | 39 |
| Positive | 0.846 | 0.723 | 0.780 | 274 |

For comparison, the TF-IDF and colour-histogram baseline scores 0.504 macro-F1 and 0.622 accuracy. Random forest and MLP scores can vary slightly between environments.
