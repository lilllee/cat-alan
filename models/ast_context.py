"""Inference wrapper for the AST-based CatMeows context classifier.

Pairs the frozen AST encoder (MIT/ast-finetuned-audioset-10-10-0.4593) with the
trained logistic-regression head in models/context_ast_head.joblib.
"""
from pathlib import Path

import numpy as np
import torch
import joblib
from scipy.signal import resample_poly
from transformers import ASTModel, AutoFeatureExtractor

from models.prosody import prosody_features

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000
HEAD_PATH = Path(__file__).resolve().parent.parent / "models" / "context_ast_head.joblib"

# Human-friendly labels for the raw class names stored with the head.
PRETTY = {
    "brushing": "Brushing",
    "isolation_unfamiliar_environment": "Isolation",
    "waiting_for_food": "Waiting for food",
}


class ASTContextClassifier:
    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.encoder = ASTModel.from_pretrained(MODEL_ID)
        self.encoder.eval()
        bundle = joblib.load(HEAD_PATH)
        self.head = bundle["classifier"]
        self.classes = bundle["classes"]
        # Older heads predate prosody; default to off so inference dims match.
        self.use_prosody = bundle.get("use_prosody", False)

    @torch.no_grad()
    def embed(self, waveform, sample_rate):
        wav = np.asarray(waveform, dtype="float32")
        if sample_rate != AST_SAMPLE_RATE:
            wav = resample_poly(wav, AST_SAMPLE_RATE, sample_rate).astype("float32")
        inputs = self.extractor(
            wav, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        hidden = self.encoder(**inputs).last_hidden_state
        return hidden.mean(dim=1).squeeze(0).numpy()

    def predict_proba(self, waveform, sample_rate):
        emb = self.embed(waveform, sample_rate)
        if self.use_prosody:
            emb = np.concatenate([emb, prosody_features(waveform, sample_rate)])
        probs = self.head.predict_proba(emb.reshape(1, -1))[0]
        return {PRETTY.get(c, c): float(p) for c, p in zip(self.classes, probs)}
