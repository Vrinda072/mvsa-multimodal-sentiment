"""Settings, optional text libraries and text-cleaning functions used by the preprocessing notebooks."""
# Imports and settings; every path can be overridden with an environment variable for local testing
import os, re, sys, json, html, hashlib, subprocess, unicodedata, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

DATA_ROOT      = os.environ.get("MVSA_ROOT", "/kaggle/input")               # where the dataset is mounted
OUT_DIR        = os.environ.get("MVSA_OUT", "/kaggle/working/preprocessed")  # where outputs are written
RUN_CLIP       = os.environ.get("MVSA_RUN_CLIP", "1") == "1"                # set to False to skip CLIP
CLIP_MODEL     = os.environ.get("MVSA_CLIP_MODEL", "openai/clip-vit-base-patch32")  # ViT-B/32
INSTALL_EXTRAS = os.environ.get("MVSA_INSTALL_EXTRAS", "1") == "1"
SEED   = 42
LABELS = ["negative", "neutral", "positive"]
WORKERS = 16          # threads for reading images in parallel (reading files is the slow part)

import socket, time
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
_T = {"t": time.perf_counter()}
def tick(step):       # prints how long each step took, so you can see where the time goes
    now = time.perf_counter()
    print(f"\n⏱ {step} took {now - _T['t']:.1f}s")
    _T["t"] = now

def online(host="pypi.org", port=443, timeout=3):
    """Check for internet access, so optional downloads can be skipped instead of hanging."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False
ONLINE = online()
socket.setdefaulttimeout(20)   # so no download can hang forever
print(f"Internet: {'on' if ONLINE else 'OFF: optional libraries and VADER will be skipped, CLIP cannot download'}")
try:
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE: CLIP will be slow. Settings → Accelerator → GPU'}")
except ImportError:
    pass

# Optional helpers: each import below falls back to None, and later code then uses a built-in alternative
if INSTALL_EXTRAS and ONLINE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "15", "--retries", "1",
                    "emoji", "wordsegment", "ftfy"], check=False)

try:
    import emoji
except ImportError:
    emoji = None
try:
    import wordsegment
    wordsegment.load()
except ImportError:
    wordsegment = None
try:
    import ftfy
except ImportError:
    ftfy = None
print(f"emoji: {emoji is not None} | wordsegment: {wordsegment is not None} | ftfy: {ftfy is not None}")
tick("Step 0 (setup)")

for sub in ("features", "artifacts", "figures"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
REPORT = {}   # every count we log ends up in preprocessing_report.json

from mvsa.state import save_state, load_state
from mvsa.text import (URL_RE, MENTION_RE, RT_RE, HASHTAG_RE, ELONG_RE, EMPTY_PLACEHOLDER, segment_hashtag,
                       is_emoji_char, count_emojis, emojis_to_text, clean_text)
