# Experiment log

Model training runs and their results, newest first. Git history tracks code
changes; this file tracks the numbers that came out of them.

Conventions:
- One entry per training/evaluation run worth remembering (skip broken runs).
- Always record: date, data (source + size), setup, and the honest metric —
  accuracy on **unseen cats / held-out samples**, not cross-validation.
- Update the README accuracy table when a model ships to the app.

---

## 2026-07-06 — Context: AST embeddings + logistic head *(shipped)*

- **Data**: CatMeows via HF `oliveirabruno01/openfarm-catmeows` (Zenodo 4008297),
  3 classes: brushing / isolation / waiting-for-food.
- **Setup**: frozen `MIT/ast-finetuned-audioset-10-10-0.4593` embeddings +
  logistic regression head (`actions/extract_ast_embeddings.py` →
  `actions/train_ast_head.py`). Head saved to `models/context_ast_head.joblib`.
- **Result**: **~59% on unseen cats** (chance 33%). CV on the training set
  gives ~72% — inflated because the model partly keys on individual cats and
  recording conditions, not context. Trust the unseen-cat number.

## 2026-07-06 — Context: raw-waveform M5 baseline *(not shipped)*

- **Data**: same CatMeows 3-class split.
- **Setup**: M5 on raw waveforms (`actions/train_context.py`).
- **Result**: ~36% on unseen cats — barely above chance. Raw waveforms don't
  transfer across cats at this data size; embeddings approach kept instead.

## (upstream) — Sentiment: M5 on raw waveforms *(shipped)*

- **Data**: CatSound (Zenodo 4724180), 10 emotion classes. Augmented with time
  stretching, pitch shifting, Gaussian noise.
- **Setup**: M5 on raw 44.1 kHz waveforms (`actions/train.py`,
  `configs/config.json`). Checkpoint at `examples/model/data/model.pth`.
- **Result**: **~58% on 50 held-out samples**. Trained by upstream
  (dogeplusplus); kept as-is in this fork.

---

## Planned

- Record my own cat per situation (food / play / angry) and train a
  cat-specific classifier. Motivation: CatMeows showed models key heavily on
  individual cats, so a single-cat model should beat both generic ones.
