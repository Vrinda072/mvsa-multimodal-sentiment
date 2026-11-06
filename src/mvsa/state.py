"""Save and load intermediate results, so that each notebook can start where the last one stopped."""
import os
import joblib

INTERIM_DIR = os.environ.get("MVSA_INTERIM", "/kaggle/working/interim")


def save_state(name, obj):
    """Save one object to the interim folder."""
    os.makedirs(INTERIM_DIR, exist_ok=True)
    joblib.dump(obj, os.path.join(INTERIM_DIR, f"{name}.joblib"))


def load_state(name):
    """Load an object saved by an earlier notebook."""
    path = os.path.join(INTERIM_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the earlier notebooks first (see README.md).")
    return joblib.load(path)
