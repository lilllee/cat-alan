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

Ordered roughly by effort. Each should land as a dated entry above when run.

1. **Add explicit prosodic features to the Context model.** Concat mean F0,
   duration, and an intonation-contour descriptor (rising / level / falling)
   onto the AST embedding before the logistic head. Schötz et al. 2023
   (70 cats, 969 meows) showed context significantly drives duration and
   *mean* F0 — but **not F0 range** — and that contour is context-specific
   (carrier=falling, cuddle/door=level, food/play/greeting=rise+fall combos).
   So prefer mean F0 + duration + contour over F0-range features.
2. **Add an MFCC + SVM baseline** (Skyler-Luo's recipe: 20 MFCC + Δ + ΔΔ, 4
   stats = 240-d, RBF SVM). A cheap third point of comparison against the AST
   head. Evaluate on the **unseen-cat split**, not the inflated 10-fold CV
   those references report.
3. **Record my own cat per situation** (food / play / angry) and train a
   single-cat classifier. Motivation: meows encode little individual identity
   but are *highly variable* between cats (Russo et al. 2025: domestic meows
   have far greater acoustic dispersion than wild felids), so a generic model
   generalizes poorly — but a **single-cat** model sidesteps individual
   variation entirely and should beat both generic models. Use the Meowsic
   protocol: video+audio, per-situation, controlled stimuli.
4. **Reach-goal architecture: JL-TFMSFNet** (Mel spectrogram + multi-scale
   feature extraction + time-frequency attention). Reported ~94–96% on its own
   datasets; a target if the feature-engineering steps above plateau.

## References

External sources analyzed for this project (2026-07-07). Kept here rather than
in CLAUDE.md so it doesn't cost context every session.

Datasets (all trace to the same two originals we already use):
- CatMeows original — Zenodo 4008297 (CC-BY-4.0). `extras.zip` has excluded /
  uncut vocalization sequences we don't currently use; filenames encode cat ID
  for honest group-by-cat splits.
- CatMeows mirrors — HF `oliveirabruno01/openfarm-catmeows` (in use) and HF
  `zeddez/CatMeows` (Parquet, tidy `label`/`audioduration` cols — backup source).
- CatSound V2 — Zenodo 4724180 (CC-BY-4.0). Public page has a 2.5 MB sample
  only; the full 12.2 GB is behind the authors' Google Drive (same data our
  dead DVC remote pointed at).

Methods / findings:
- Schötz et al. 2023, *Applied Animal Behaviour Science* — context effects on
  meow duration/F0/intonation, 70 cats / 969 meows / 7 contexts.
  https://www.sciencedirect.com/science/article/pii/S0168159123003180
- Russo, Schild & Knörnschild 2025, *Scientific Reports* — meows encode less
  individual info than purrs (64% vs 85% DFA identity accuracy), domestic
  meows more variable than wild. Open access:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12695941/
- JL-TFMSFNet 2024, *Expert Systems with Applications* —
  https://dl.acm.org/doi/10.1016/j.eswa.2024.124620
- Pandeya et al. 2018 (CatSound origin) — https://www.mdpi.com/2076-3417/8/10/1949
- Skyler-Luo/CatMeows-Recognition (MFCC+SVM baseline) —
  https://github.com/Skyler-Luo/CatMeows-Recogintion
- Meowsic project (Schötz/Eklund, Lund) — recording protocol + F0-contour
  perception results. https://meowsic.se/resultsMeowing.html
