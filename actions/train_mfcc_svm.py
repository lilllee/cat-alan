"""MFCC + SVM baseline for CatMeows context classification.

A third comparison point alongside the deployed AST-embedding logistic head and
the raw-waveform M5, following the recipe in Skyler-Luo/CatMeows-Recognition:
20 MFCCs + Δ + ΔΔ, aggregated over frames with 4 statistics (mean/std/min/max)
= 240-d, classified by an RBF SVM (C=10, gamma=scale).

Reported on the unseen-cat test split — NOT the 10-fold CV the reference uses,
which mixes each cat across folds and inflates accuracy. This baseline is for
comparison only; it is not saved or wired into the app.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]
SRC_SAMPLE_RATE = 8000
N_MFCC = 20
N_MELS = 26
N_FFT = 256          # ~32 ms at 8 kHz (approximates the 30 ms spec)
HOP = 160            # 20 ms
FMAX = 4000          # Nyquist at 8 kHz


def mfcc_features(wav, sr):
    """20 MFCC + Δ + ΔΔ (60 per frame), pooled over time into 240 stats."""
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP, fmax=FMAX,
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    frames = np.vstack([mfcc, d1, d2])  # (60, T)
    stats = np.concatenate([
        frames.mean(axis=1), frames.std(axis=1),
        frames.min(axis=1), frames.max(axis=1),
    ])
    return stats.astype("float32")


def load_split(split_dir):
    feats, labels = [], []
    for label, name in enumerate(CLASSES):
        for f in sorted(Path(split_dir, name).glob("*.wav")):
            wav, sr = sf.read(f, dtype="float32")
            assert sr == SRC_SAMPLE_RATE, f"{f}: unexpected sample rate {sr}"
            feats.append(mfcc_features(wav, sr))
            labels.append(label)
    return np.stack(feats), np.array(labels)


def main():
    root = Path("data/catmeows_context")
    xtr, ytr = load_split(root / "train")
    xte, yte = load_split(root / "test")
    print(f"train {xtr.shape}  test {xte.shape}")

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print(f"5-fold CV on train (inflated): {cross_val_score(clf, xtr, ytr, cv=cv).mean():.2%}")

    clf.fit(xtr, ytr)
    print(f"train accuracy:           {accuracy_score(ytr, clf.predict(xtr)):.2%}")
    print(f"test accuracy (new cats): {accuracy_score(yte, clf.predict(xte)):.2%}")
    print("confusion matrix (rows=true, cols=pred):", CLASSES)
    print(confusion_matrix(yte, clf.predict(xte)))


if __name__ == "__main__":
    main()
