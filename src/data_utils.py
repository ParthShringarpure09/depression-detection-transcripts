"""
data_utils.py
Shared data-loading utilities for the depression-detection text branch.

Loaded once, reused across every classification experiment (SIL, MIL, LR, RNN).
Nothing experiment-specific lives here — no model architectures, no training loops.

Design principle: read-only helpers over the pre-existing features/ and
fold-lists.csv. Nothing here modifies disk state.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- paths (relative to src/) ----------

FOLD_CSV_PATH = Path("../data/Androids-Corpus/fold-lists.csv")
SEGMENTS_DIR = Path("../segments")
FEATURES_DIR = Path("../features")


# ---------- fold assignment ----------

def _build_speaker_to_fold():
    """
    Parse fold-lists.csv (Interview-Task columns 7..11) into
    {speaker_stem: fold_number}. Called once at import time.
    """
    raw = pd.read_csv(FOLD_CSV_PATH, header=None, skiprows=2)
    interview_cols = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11}

    mapping = {}
    for fold_num, col_idx in interview_cols.items():
        for speaker in raw[col_idx].dropna():
            stem = str(speaker).strip().strip("'")
            mapping[stem] = fold_num
    return mapping


speaker_to_fold = _build_speaker_to_fold()


# ---------- filename parsing ----------

_STEM_SUFFIX_RE = re.compile(r"_N\d+_seg\d+$")

def speaker_from_filename(path: Path) -> str:
    """
    Extract the speaker stem from a segment or feature filename.
    E.g. '01_CF56_1_N8_seg3.npy' -> '01_CF56_1'.
    """
    return _STEM_SUFFIX_RE.sub("", path.stem)


# ---------- feature loading ----------

def load_data_for_n(n: int):
    """
    Load all feature vectors for a given N-level, along with their
    label (0=HC/control, 1=PT/depressed), fold number, and speaker id.

    Returns:
        X       : ndarray of shape (num_segments, 768)
        y       : ndarray of shape (num_segments,)   — labels in {0, 1}
        fold    : ndarray of shape (num_segments,)   — fold in {1..5}
        speaker : ndarray of shape (num_segments,)   — speaker stems (strings)
    """
    X, y, fold, speaker = [], [], [], []

    for label_name, label_value in [("HC", 0), ("PT", 1)]:
        pattern = f"*_N{n}_seg*.npy"
        for f in sorted((FEATURES_DIR / label_name).glob(pattern)):
            vec = np.load(f)
            spk = speaker_from_filename(f)

            X.append(vec)
            y.append(label_value)
            fold.append(speaker_to_fold[spk])
            speaker.append(spk)

    return np.array(X), np.array(y), np.array(fold), np.array(speaker)


# ---------- train/test splitting ----------

def get_fold_split(X, y, fold, speaker, fold_number: int):
    """
    Person-independent train/test split: hold out one fold as test,
    train on the other four.

    Returns X_train, y_train, X_test, y_test, speaker_test
    (speaker_test is needed downstream for majority-vote aggregation.)
    """
    test_mask = (fold == fold_number)
    train_mask = ~test_mask

    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask], speaker[test_mask],
    )


# ---------- misc constants ----------

N_LEVELS = [1, 2, 4, 8, 16, 32, 64]