# Methodology

This page explains the steps in preprocessing, training and evaluation. The notebooks hold the full detail and code.

## Preprocessing

1. **Missing and broken data.** We checked for invalid labels, missing or empty text, missing or corrupt images, and exact duplicates. The 30 duplicates were the only problem found. Raw text and pixels are never imputed.
2. **Labels.** Matching text and image labels give that label. If one label is neutral, the other is used (a soft mismatch). Positive against negative has no single label (a hard mismatch), so these posts are held out of training.
3. **Text cleaning.** HTML entities are decoded, URLs and retweet prefixes removed, mentions replaced with `@user`, hashtags split into words, emojis converted to words and elongated words shortened. Case, punctuation and stop-words stay, because they can signal sentiment. Standard stop-word lists include "not".
4. **Splitting.** Labelled posts are split 8:1:1 before anything is fitted. The split is stratified on label and agreement group, and identical captions stay in one split, so a retweeted caption can't leak into the test set.
5. **Features.** These are frozen CLIP ViT-B/32 text and image embeddings (512 dimensions each) and their cosine similarity, TF-IDF on word 1-2-grams, an 8-bin colour histogram per channel, and engineered text and image features.
6. **Scaling.** Counts and sizes are log-transformed and standardised, and ratios are standardised, using training-set statistics only. Binary flags stay 0 or 1. Histograms are normalised per channel, and TF-IDF and CLIP vectors to unit length.

## Models

Classical classifiers are trained on frozen CLIP embeddings rather than training deep networks end to end. CLIP has about 150 million parameters, so fine-tuning it on 3,591 noisy posts would risk overfitting. Frozen features also make training cheap enough to repeat over five seeds.

| Classifier | Why it is included | Key settings |
|---|---|---|
| Logistic regression | Standard linear probe for CLIP, with direct probabilities and readable coefficients | Balanced class weights; `C` tuned |
| Linear SVM | Margin-based and robust in high dimensions | Balanced class weights; `C` tuned; Platt scaling for probabilities |
| Random forest | Tests whether tree-based non-linearity helps | Balanced subsample weights; 300 trees; minimum leaf size tuned |
| MLP (256 hidden units) | Tests whether learned text-image interactions help | `alpha` tuned; early stopping; minority classes oversampled |

| Fusion setup | Input |
|---|---|
| Baseline | TF-IDF and colour histogram, with logistic regression |
| Text only / image only | One CLIP embedding |
| Early fusion | Concatenated text and image embeddings |
| Early + meta | Early fusion plus engineered features |
| Late fusion | `w × p_text + (1 − w) × p_image`, with `w` tuned |
| Gated fusion (proposed) | A stacking logistic regression on both sets of probabilities, CLIP cosine similarity, Jensen-Shannon divergence and each classifier's confidence |

The gate trains on 5-fold out-of-fold predictions, which stops it learning from over-confident probabilities on posts the single-modality models have already seen.

## Training protocol

1. Tune hyperparameters on the validation set over small grids, using seed 0.
2. Tune the late-fusion weight and the gate's regularisation on the same set.
3. Retrain each tuned setup over five seeds and report the mean and standard deviation.
4. Choose the final model by validation macro-F1. Test scores are reported but never used to choose.

## Evaluation

- **Accuracy:** the share of all posts predicted correctly.
- **Precision:** of the posts predicted as a class, the share that really are in it.
- **Recall:** of the posts really in a class, the share the model found.
- **F1:** the harmonic mean of precision and recall.
- **Macro averages:** give all three classes equal weight. Macro-F1 drives model selection because the neutral class is small.

We also report accuracy per agreement group, which modality the model follows on hard-mismatch posts, a McNemar test between early and gated fusion, and scores with one modality masked by its training-set mean embedding.
