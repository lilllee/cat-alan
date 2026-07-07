"""Extract AST (Audio Spectrogram Transformer) embeddings for the CatMeows
context clips and cache them to disk.

AST expects 16kHz audio; the source wavs are 8kHz so they are resampled first.
The frozen encoder's mean-pooled last hidden state (768-d) is used as a fixed
feature for a lightweight downstream classifier.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import ASTModel, AutoFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000
SRC_SAMPLE_RATE = 8000


def iter_clips(split_dir):
    for label, name in enumerate(CLASSES):
        for f in sorted(Path(split_dir, name).glob("*.wav")):
            data, sr = sf.read(f, dtype="float32")
            assert sr == SRC_SAMPLE_RATE, f"{f}: unexpected sample rate {sr}"
            data = resample_poly(data, AST_SAMPLE_RATE, SRC_SAMPLE_RATE).astype("float32")
            yield data, label


@torch.no_grad()
def embed_split(split_dir, extractor, model):
    feats, labels = [], []
    for data, label in iter_clips(split_dir):
        inputs = extractor(
            data, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        hidden = model(**inputs).last_hidden_state  # (1, tokens, 768)
        feats.append(hidden.mean(dim=1).squeeze(0).numpy())
        labels.append(label)
    return np.stack(feats), np.array(labels)


def main():
    data_root = Path("data/catmeows_context")
    out_dir = Path("data/ast_embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTModel.from_pretrained(MODEL_ID)
    model.eval()

    for split in ["train", "test"]:
        feats, labels = embed_split(data_root / split, extractor, model)
        np.savez(out_dir / f"{split}.npz", features=feats, labels=labels)
        print(f"{split}: {feats.shape} embeddings saved")


if __name__ == "__main__":
    main()
