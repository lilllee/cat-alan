"""Train a 3-class meow context classifier (waiting_for_food /
isolation_unfamiliar_environment / brushing) on the CatMeows dataset.

Expects data/catmeows_context/{train,test}/<class>/*.wav at 8kHz mono.
Saves the best state_dict to models/context_m5.pth.
"""
import sys
import copy
import random
import argparse

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf

from models.m5 import M5

CLASSES = ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]
SAMPLE_RATE = 8000


def load_split(split_dir):
    items = []
    for label, name in enumerate(CLASSES):
        for f in sorted(Path(split_dir, name).glob("*.wav")):
            data, sr = sf.read(f, dtype="float32")
            assert sr == SAMPLE_RATE, f"{f}: unexpected sample rate {sr}"
            items.append((data, label))
    return items


def augment(wav, rng):
    wav = wav * rng.uniform(0.6, 1.4)
    if rng.random() < 0.5:
        wav = wav + rng.normal(0, rng.uniform(0.001, 0.01), wav.shape).astype("float32")
    if rng.random() < 0.5:
        wav = np.roll(wav, rng.integers(-SAMPLE_RATE // 4, SAMPLE_RATE // 4))
    return wav


def batches(items, batch_size, rng=None, train=False):
    order = list(range(len(items)))
    if train:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[start:start + batch_size]]
        wavs = [augment(w, rng) if train else w for w, _ in chunk]
        max_len = max(len(w) for w in wavs)
        features = torch.zeros((len(chunk), 1, max_len))
        for i, w in enumerate(wavs):
            features[i, 0, :len(w)] = torch.from_numpy(np.ascontiguousarray(w))
        labels = torch.LongTensor([l for _, l in chunk])
        yield features, labels


def evaluate(model, items, batch_size=16):
    model.eval()
    correct = 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    with torch.no_grad():
        for wavs, labels in batches(items, batch_size):
            preds = torch.argmax(model(wavs)[:, 0], dim=-1)
            correct += (preds == labels).sum().item()
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1
    return correct / len(items), confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/catmeows_context")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/context_m5.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_items = load_split(Path(args.data, "train"))
    test_items = load_split(Path(args.data, "test"))
    random.shuffle(train_items)
    val_size = len(train_items) // 5
    val_items, train_items = train_items[:val_size], train_items[val_size:]
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    model = M5(num_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        running_loss, samples = 0.0, 0
        for wavs, labels in batches(train_items, args.batch_size, rng, train=True):
            optimizer.zero_grad()
            loss = loss_fn(model(wavs)[:, 0], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.shape[0]
            samples += labels.shape[0]

        val_acc, _ = evaluate(model, val_items)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d} loss {running_loss / samples:.4f} "
                  f"val_acc {val_acc:.2%} (best {best_val:.2%})")

    model.load_state_dict(best_state)
    test_acc, confusion = evaluate(model, test_items)
    print(f"\nbest val accuracy: {best_val:.2%}")
    print(f"test accuracy: {test_acc:.2%}")
    print("confusion matrix (rows=true, cols=pred):", CLASSES)
    print(confusion)

    torch.save(best_state, args.out)
    print(f"saved state_dict to {args.out}")


if __name__ == "__main__":
    main()
