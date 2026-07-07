import subprocess

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from tempfile import TemporaryDirectory

from imageio_ffmpeg import get_ffmpeg_exe

from models.m5 import M5
from models.ast_context import ASTContextClassifier

SENTIMENT = "Sentiment (10 emotions)"
CONTEXT = "Context (why is it meowing?)"

# The sentiment M5 was trained on 44.1kHz audio loaded at its native rate, so
# inputs must be resampled to match before inference.
SENTIMENT_SAMPLE_RATE = 44100

SENTIMENT_CLASSES = {
    0: "Happy",
    1: "Resting",
    2: "Angry",
    3: "Paining",
    4: "Mother Call",
    5: "Warning",
    6: "Hunting",
    7: "Fighting",
    8: "Defence",
    9: "Mating",
}

SENTIMENT_EMOJIS = {
    "Happy": "🐱 : Cat is happy.",
    "Resting": "💤 : Cat is tired.",
    "Angry": "😾 : Cat is angry.",
    "Paining": "😿 : Cat sounds like it's in pain.",
    "Mother Call": "🙀 : Cat is calling for mum.",
    "Warning": "⚠️ : Cat is giving you a warning.",
    "Hunting": "😼 : Cat wants to hunt.",
    "Fighting": "⚔️ : Cat is about to throw hands.",
    "Defence": "🛡️  : Cat is on the defence.",
    "Mating": "😻 : Cat wants to mate.",
}

CONTEXT_EMOJIS = {
    "Brushing": "🪮 : Cat is being brushed (mild annoyance).",
    "Isolation": "🙀 : Cat is stressed / alone in an unfamiliar place.",
    "Waiting for food": "🍚 : Cat wants food!",
}


@st.cache_resource
def load_sentiment_model():
    # The checkpoint is a full pickled M5 module; allowlist only the classes it
    # needs so it loads under the safe weights-only unpickler.
    torch.serialization.add_safe_globals([
        M5, nn.Sequential, nn.Conv1d, nn.BatchNorm1d, nn.ReLU,
        nn.MaxPool1d, nn.Dropout, nn.Linear, nn.LogSoftmax,
    ])
    model = torch.load(
        "examples/model/data/model.pth", map_location="cpu", weights_only=True
    )
    model.eval()
    return model


@st.cache_resource
def load_context_model():
    return ASTContextClassifier()


def to_wav(input_path, output_path, sample_rate):
    subprocess.run(
        [
            get_ffmpeg_exe(), "-y", "-i", str(input_path),
            "-ac", "1", "-ar", str(sample_rate), str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def read_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def sentiment_probs(model, waveform):
    audio = torch.from_numpy(waveform).reshape(1, 1, -1)
    with torch.no_grad():
        log_softmax = model(audio)
    probs = torch.squeeze(torch.exp(log_softmax)).cpu().numpy()
    return {SENTIMENT_CLASSES[i]: float(p) for i, p in enumerate(probs)}


def render(scores, emojis, ylabel):
    top = max(scores, key=scores.get)
    left.subheader(emojis[top])
    fig, ax = plt.subplots()
    pairs = sorted(scores.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in pairs], [v for _, v in pairs])
    ax.set_xlabel("Probability")
    ax.set_ylabel(ylabel)
    left.pyplot(fig)


st.title("Meow Sentiment Analysis")
model_choice = st.sidebar.selectbox("Model", [SENTIMENT, CONTEXT])
st.sidebar.subheader("Provide Audio or Video file")
uploaded_file = st.sidebar.file_uploader("File Path", type=["mp4", "mp3", "wav", "m4a"])
recorded_audio = st.sidebar.audio_input("...or record a meow")
if model_choice == SENTIMENT:
    st.sidebar.write("10-emotion classifier (M5 on raw waveforms, CatSound dataset).")
else:
    st.sidebar.write("3-context classifier (AST embeddings + logistic head, CatMeows dataset). ~59% accuracy on unseen cats.")
st.sidebar.write("**For optimal results try to keep the audio to just the meow.")

left, right = st.columns([2, 1])

source = uploaded_file if uploaded_file is not None else recorded_audio

if source is not None:
    raw_bytes = source.read()
    extension = Path(source.name).suffix if uploaded_file is not None else ".wav"

    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir, f"input{extension}")
        with open(input_path, "wb") as f:
            f.write(raw_bytes)

        if extension == ".mp4":
            right.video(raw_bytes)
        else:
            right.audio(raw_bytes)

        if model_choice == SENTIMENT:
            wav_path = Path(temp_dir, "audio.wav")
            to_wav(input_path, wav_path, SENTIMENT_SAMPLE_RATE)
            waveform, _ = read_mono(wav_path)
            scores = sentiment_probs(load_sentiment_model(), waveform)
            render(scores, SENTIMENT_EMOJIS, "Sentiment")
        else:
            # AST handles resampling internally; just decode to a wav it can read.
            wav_path = Path(temp_dir, "audio.wav")
            to_wav(input_path, wav_path, 16000)
            waveform, sr = read_mono(wav_path)
            scores = load_context_model().predict_proba(waveform, sr)
            render(scores, CONTEXT_EMOJIS, "Context")
