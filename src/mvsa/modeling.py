"""Data loading, classifiers, fusion setups, tuning and evaluation used by the modelling notebooks.

The modelling notebooks start with `from mvsa.modeling import *`. Importing this module loads every split of
the preprocessed features into D, so the notebooks can go straight to training. Sections:
  1. settings and data loading          4. fusion setups and the agreement gate
  2. classifier families and grids      5. hyperparameter tuning on the validation set
  3. training helpers                   6. evaluation metrics
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
# The MLP and SVM sometimes stop at their iteration limit; the warning is noise because results are checked on
# validation data anyway
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---- Settings and data loading
QUICK    = os.environ.get("MVSA_QUICK", "0") == "1"   # True: 1 seed, for a fast test run
SEEDS    = [0] if QUICK else [0, 1, 2, 3, 4]           # every reported score is the mean over these seeds
OUT_DIR  = os.environ.get("MVSA_MODEL_OUT", "/kaggle/working/models")
LABELS   = ["negative", "neutral", "positive"]         # class i is LABELS[i] in every array
FAMILIES = ["logreg", "linear_svm", "random_forest", "mlp"]   # the four classifiers compared
os.makedirs(OUT_DIR, exist_ok=True)

def find_preprocessed(roots=("/kaggle/working", "/kaggle/input")):
    """Locate the preprocessing notebook's output folder, wherever it was attached."""
    if os.environ.get("MVSA_PRE"):
        return os.environ["MVSA_PRE"]
    for r in roots:
        for dirpath, dirs, files in os.walk(r):
            if "posts_preprocessed.csv" in files and "features" in dirs:
                return dirpath
    raise FileNotFoundError("preprocessed/ not found. Run the notebooks in notebooks/01_preprocessing first, or set MVSA_PRE.")

PRE = find_preprocessed()
F = os.path.join(PRE, "features")
# One row per post with its labels and agreement group; ids are read as text so "0012" keeps its zeros
posts = pd.read_csv(os.path.join(PRE, "posts_preprocessed.csv"), dtype={"id": str}).set_index("id")
META_COLS = json.load(open(os.path.join(F, "meta_columns.json")))   # names of the engineered meta features

def load(split):
    """Load one split. Row i of every array belongs to the post on line i of <split>_ids.txt."""
    ids = np.loadtxt(os.path.join(F, f"{split}_ids.txt"), dtype=str, ndmin=1)
    p = posts.loc[ids]
    return {
        "ids": ids,
        "text":  np.load(os.path.join(F, f"{split}_clip_text.npy")),    # CLIP caption embeddings, 512 per post
        "image": np.load(os.path.join(F, f"{split}_clip_image.npy")),   # CLIP image embeddings, 512 per post
        "meta":  np.load(os.path.join(F, f"{split}_meta.npy")),         # engineered features (counts, ratios, flags)
        "hist":  np.load(os.path.join(F, f"{split}_colorhist.npy")),    # colour histogram, for the baseline only
        "tfidf": sparse.load_npz(os.path.join(F, f"{split}_tfidf.npz")),  # TF-IDF word counts, for the baseline
        "y":     p["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy(),   # NaN for 'hard'
        "group": p["mismatch_type"].to_numpy(),
        "text_label": p["text_label"].to_numpy(), "image_label": p["image_label"].to_numpy(),
    }

# "hard" holds the hard-mismatch posts: they have no single label, so they are never trained on
D = {s: load(s) for s in ["train", "val", "test", "hard"]}
for s in ["train", "val", "test"]:
    D[s]["y"] = D[s]["y"].astype(int)   # these splits always have a label, so whole-number classes are safe
ytr = D["train"]["y"]                   # shortcut for the training labels
print(f"Loaded from {PRE}")
for s, d in D.items():
    print(f"  {s:5s} {len(d['ids']):5d} posts | CLIP text {d['text'].shape[1]}-d, image {d['image'].shape[1]}-d")
print("Train class counts:", dict(zip(LABELS, np.bincount(ytr, minlength=3))))

from mvsa.state import save_state, load_state


# ---- Classifier families and tuning grids
# Small grids on purpose: a large search could overfit the 446-post validation set
GRIDS = {
    "logreg":        [{"C": c} for c in (0.01, 0.1, 1.0)],
    "linear_svm":    [{"C": c} for c in (0.001, 0.01, 0.1)],
    "random_forest": [{"min_samples_leaf": m} for m in (1, 5)],
    "mlp":           [{"alpha": a} for a in (1e-3, 1e-1, 1.0)],
}

def make_clf(family, seed, C=1.0, min_samples_leaf=1, alpha=1e-3):
    """Build one untrained classifier. Scalers sit inside the pipeline, so they are fitted on training data only."""
    if family == "logreg":                 # C is the inverse of the L2 penalty: smaller C = stronger regularisation
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    if family == "linear_svm":             # LinearSVC has no probabilities, so Platt scaling ("sigmoid") adds them
        return make_pipeline(StandardScaler(),
                             CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", max_iter=20000),
                                                    method="sigmoid", cv=3))
    if family == "random_forest":          # a larger minimum leaf size gives simpler trees
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if family == "mlp":                    # alpha is the L2 penalty; early stopping also limits overfitting
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(256,), alpha=alpha, early_stopping=True,
                                           validation_fraction=0.15, max_iter=500, random_state=seed))
    raise ValueError(family)

# ---- Training helpers
def oversample(X, y, seed):
    """Repeat minority-class rows until every class is as large as the biggest (used for the MLP only)."""
    rng = np.random.default_rng(seed)
    n = np.bincount(y).max()                    # size of the largest class
    # For each class, draw n row numbers with replacement, so small classes get repeated rows
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=True) for c in np.unique(y)])
    return X[idx], y[idx]

def fit(family, hp, X, y, seed):
    """Train one classifier. The MLP has no class weights, so it gets oversampled data instead."""
    if family == "mlp":
        X, y = oversample(X, y, seed)
    return make_clf(family, seed, **hp).fit(X, y)

def oof_proba(family, hp, X, y, seed, k=5):
    """Out-of-fold probabilities: each row is predicted by a model that never saw it (A5, point 1)."""
    P = np.zeros((len(y), len(LABELS)))
    # Split into k folds; train on k-1 of them and predict the fold left out, so every row gets one prediction
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        P[te] = fit(family, hp, X[tr], y[tr], seed).predict_proba(X[te])
    return P


# ---- Fusion setups and the agreement gate
def features(d, kind):
    """Return the input matrix for one kind of model (each lambda runs only when it is selected)."""
    return {"text":       lambda: d["text"],
            "image":      lambda: d["image"],
            "early":      lambda: np.hstack([d["text"], d["image"]]),
            "early_meta": lambda: np.hstack([d["text"], d["image"], d["meta"]])}[kind]()

# Names of the gate's 10 input columns, in order (used to label its coefficients in the analysis)
GATE_COLS = ([f"p_text_{l}" for l in LABELS] + [f"p_image_{l}" for l in LABELS]
             + ["clip_cosine", "js_divergence", "conf_text", "conf_image"])

def gate_inputs(p_text, p_image, d):
    """Build the gate's 10 inputs: both sets of class probabilities plus 4 agreement signals."""
    cos = (d["text"] * d["image"]).sum(axis=1, keepdims=True)   # embeddings are unit length, so this is the cosine
    js  = np.nan_to_num(jensenshannon(p_text, p_image, axis=1, base=2))[:, None]   # 0 = agree, 1 = disagree
    # Confidence of each model = its highest class probability
    return np.hstack([p_text, p_image, cos, js, p_text.max(1, keepdims=True), p_image.max(1, keepdims=True)])

def make_gate(C):
    """The gate itself: a logistic regression that learns how to combine the 10 inputs into one prediction."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))

def macro_f1(y, P):
    """Unweighted mean of the per-class F1 scores, so the small neutral class counts as much as the others."""
    return f1_score(y, P.argmax(1), average="macro")

def build(family, cfg, seed, train=D["train"]):
    """Train every setup for one family and seed. Returns {setup: predict_proba function} and the gate."""
    hp = cfg["hp"]
    # One model per input type, trained on the full training set
    m = {k: fit(family, hp[k], features(train, k), train["y"], seed) for k in ("text", "image", "early", "early_meta")}
    # The gate trains on out-of-fold probabilities, so it never sees over-confident scores on already-seen posts
    Pt = oof_proba(family, hp["text"],  train["text"],  train["y"], seed)
    Pv = oof_proba(family, hp["image"], train["image"], train["y"], seed)
    gate = make_gate(cfg["gate_C"]).fit(gate_inputs(Pt, Pv, train), train["y"])
    w = cfg["late_w"]                                     # late-fusion weight on the text model, tuned on validation
    pt = lambda d: m["text"].predict_proba(d["text"])     # text-only probabilities for any split d
    pv = lambda d: m["image"].predict_proba(d["image"])   # image-only probabilities for any split d
    # Each entry turns a split into class probabilities, so every setup is scored with the same code
    return {
        "text_only":  pt,
        "image_only": pv,
        "early":      lambda d: m["early"].predict_proba(features(d, "early")),
        "early_meta": lambda d: m["early_meta"].predict_proba(features(d, "early_meta")),
        "late":       lambda d: w * pt(d) + (1 - w) * pv(d),
        "gated":      lambda d: gate.predict_proba(gate_inputs(pt(d), pv(d), d)),
    }, gate


# ---- Hyperparameter tuning on the validation set
def tune(family, seed=0, train=D["train"], val=D["val"]):
    """Choose one family's settings by validation macro-F1: grid values, the late-fusion weight and the gate's C.

    Every model is trained on the training split and scored on the validation split; the test split is never used.
    """
    hp, scores = {}, {}
    # Step 1: best grid value for each input type
    for kind in ("text", "image", "early", "early_meta"):
        # Train once per grid value and keep the one with the highest validation macro-F1
        best = max(GRIDS[family], key=lambda g: macro_f1(
            val["y"], fit(family, g, features(train, kind), train["y"], seed).predict_proba(features(val, kind))))
        hp[kind] = best
    # Step 2: best late-fusion weight w, from 0 (image only) to 1 (text only)
    mt = fit(family, hp["text"],  train["text"],  train["y"], seed)
    mv = fit(family, hp["image"], train["image"], train["y"], seed)
    pt, pv = mt.predict_proba(val["text"]), mv.predict_proba(val["image"])
    late_w = max(np.round(np.arange(0, 1.01, 0.1), 1), key=lambda w: macro_f1(val["y"], w * pt + (1 - w) * pv))
    # Step 3: best regularisation strength for the gate. Its training inputs are out-of-fold probabilities,
    # exactly as in build(); its validation inputs come from the models trained on all training posts
    Gtr = gate_inputs(oof_proba(family, hp["text"], train["text"], train["y"], seed),
                      oof_proba(family, hp["image"], train["image"], train["y"], seed), train)
    Gva = gate_inputs(pt, pv, val)
    gate_C = max((0.1, 1.0, 10.0), key=lambda C: macro_f1(val["y"], make_gate(C).fit(Gtr, train["y"]).predict_proba(Gva)))
    return {"hp": hp, "late_w": float(late_w), "gate_C": gate_C}


# ---- Evaluation
def evaluate(P, d):
    """Score the class probabilities P on one split, returning a dictionary of metrics."""
    pred = P.argmax(1)                                             # the predicted class is the most probable one
    if np.isnan(d["y"].astype(float)).all():                       # the 'hard' set
        # Hard mismatches have no post label, so report which part's label the prediction agrees with instead
        lab = np.array(LABELS)[pred]
        return {"matches_text": (lab == d["text_label"]).mean(),
                "matches_image": (lab == d["image_label"]).mean(),
                "pred_neutral": (lab == "neutral").mean()}
    y = d["y"]
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
           "macro_recall": recall_score(y, pred, average="macro", zero_division=0)}
    # Per-class scores: macro averages can hide one very weak class
    for name, metric in (("f1", f1_score), ("precision", precision_score), ("recall", recall_score)):
        kw = {} if name == "f1" else {"zero_division": 0}
        out.update({f"{name}_{l}": v for l, v in zip(LABELS, metric(y, pred, average=None, labels=[0, 1, 2], **kw))})
    # Accuracy per agreement group shows whether mismatched posts are harder
    for g in ("agreement", "soft_mismatch"):
        msk = d["group"] == g
        if msk.any():
            out[f"acc_{g}"] = accuracy_score(y[msk], pred[msk])
    return out
