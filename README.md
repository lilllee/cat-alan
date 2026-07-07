# cat-alan

Classification of audio files of domestic cats to 1 of 10 intents. (Warning, Angry, Defence, Fighting, Happy, Hunting, Mating, Mother Call, Paining, Resting).
Model trains on the raw waveforms using the M5 architecture written in PyTorch. Applied time stretching, pitch shifting and Gaussian noise as augmentation.

![Example](/examples/app_demo.png)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app supports file upload (wav/mp3/mp4/m4a) and in-browser microphone
recording, and offers two models from the sidebar:

| Model | Classes | Approach | Test accuracy |
|---|---|---|---|
| Sentiment | 10 emotions | M5 on raw 44.1kHz waveforms (CatSound) | ~58% on 50 held-out samples |
| Context | brushing / isolation / waiting-for-food | frozen AST embeddings + logistic regression (CatMeows) | ~59% on unseen cats (chance 33%) |

## Context model (why is the cat meowing?)

Trained on [OpenFARM CatMeows](https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows)
(derived from [CatMeows, Zenodo 4008297](https://zenodo.org/records/4008297), CC-BY-4.0).
To reproduce:

```bash
python actions/prepare_catmeows.py        # download + export wavs
python actions/extract_ast_embeddings.py  # cache frozen AST features
python actions/train_ast_head.py          # train + evaluate the classifier head
python actions/train_context.py           # (optional) raw-waveform M5 baseline, ~36%
```

Note: CatMeows' test split contains cats unseen during training, so
cross-validation on the training set (~72%) overestimates accuracy on new
cats (~59%) — the model partly keys on individual cats and recording
conditions, not just context.

## Credits

Credit to the origin of the dataset and augmentation techniques is given to:

**Domestic Cat Sound Classification Using Transfer Learning**
*Yagya Raj Pandeya, Dongwhoon Kim and Joonwhoan Lee*
*https://doi.org/10.3390/app8101949*

**CatMeows: A Publicly-Available Dataset of Cat Vocalizations**
*Ludovico, Ntalampiras, Presti et al.*
*https://doi.org/10.5281/zenodo.4008297*
