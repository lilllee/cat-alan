"""Extract prosodic features (duration, mean F0, intonation slope) for the
CatMeows context clips and cache them alongside the AST embeddings.

Reads the same wavs as extract_ast_embeddings.py but at their native rate (no
resampling — F0 estimation is more accurate on the original signal). Output
aligns row-for-row with data/ast_embeddings/{split}.npz so train_ast_head.py
can concatenate the two feature sets.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.prosody import prosody_features  # noqa: E402

CLASSES = ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]


def iter_clips(split_dir):
    # Same traversal order as extract_ast_embeddings.iter_clips so rows align.
    for label, name in enumerate(CLASSES):
        for f in sorted(Path(split_dir, name).glob("*.wav")):
            data, sr = sf.read(f, dtype="float32")
            yield data, sr, label


def main():
    data_root = Path("data/catmeows_context")
    out_dir = Path("data/ast_embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        feats, labels = [], []
        for data, sr, label in iter_clips(data_root / split):
            feats.append(prosody_features(data, sr))
            labels.append(label)
        feats, labels = np.stack(feats), np.array(labels)
        np.savez(out_dir / f"{split}_prosody.npz", features=feats, labels=labels)
        print(f"{split}: {feats.shape} prosody features saved")


if __name__ == "__main__":
    main()
