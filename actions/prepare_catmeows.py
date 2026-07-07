"""Download the OpenFARM CatMeows context dataset from the HuggingFace Hub and
export it to data/catmeows_context/{train,test}/<context>/*.wav.

Source: https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows
(derived from CatMeows, Zenodo 4008297, CC-BY-4.0). 3 contexts: brushing,
isolation_unfamiliar_environment, waiting_for_food.
"""
import io
from pathlib import Path

import pandas as pd
import soundfile as sf

BASE = "https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows/resolve/main/data"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def main():
    out_root = Path("data/catmeows_context")
    for split, fname in FILES.items():
        df = pd.read_parquet(f"{BASE}/{fname}")
        print(f"{split}: {len(df)} rows; {df['context'].value_counts().to_dict()}")
        for i, row in df.iterrows():
            label_dir = out_root / split / row["context"]
            label_dir.mkdir(parents=True, exist_ok=True)
            audio = row["audio"]
            raw = audio["bytes"] if isinstance(audio, dict) else audio
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            sf.write(label_dir / f"{row['cat_id']}_{i:04d}.wav", data, sr)
    print(f"exported to {out_root}")


if __name__ == "__main__":
    main()
