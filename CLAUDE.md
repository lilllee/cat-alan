# cat-alan

Streamlit app that classifies cat meow audio. Fork of `dogeplusplus/cat-alan`
(upstream is archived work; this fork modernized the app and added a second model).
See `@README.md` for the user-facing overview and model accuracy table.

## Environment & commands

- Python env is `.venv/` (uv-managed, **CPU-only torch**). Always use the venv
  binaries directly: `.venv/bin/python`, `.venv/bin/streamlit`.
  Never reinstall torch with CUDA — this project runs inference on CPU only.
- Run the app: `.venv/bin/streamlit run app.py` (serves on port 8501).
- Install a new dependency: `uv pip install --python .venv/bin/python <pkg>`,
  then add it to `requirements.txt` under the matching comment section.
- `tests/` contains only a placeholder and pytest is not installed in the venv;
  `make test` fails as-is. Verify changes by running the app, not the test suite.

## Architecture

Two independent models selectable from the app sidebar:

| Model | Code | Weights | Input |
|---|---|---|---|
| Sentiment (10 emotions) | `models/m5.py` | `examples/model/data/model.pth` | raw waveform, must be resampled to 44.1 kHz |
| Context (3 classes) | `models/ast_context.py` | `models/context_ast_head.joblib` | 16 kHz wav; AST encoder downloaded from HF at first run |

- `app.py` — the whole UI. Decodes any upload (wav/mp3/mp4/m4a) or in-browser
  recording to mono wav via `imageio_ffmpeg` before inference.
- The sentiment checkpoint is a **full pickled M5 module**, loaded with
  `weights_only=True` plus `add_safe_globals` in `app.py`. If you change layers
  in `models/m5.py`, update that allowlist too.
- Model loading is wrapped in `@st.cache_resource` — keep it that way, cold
  loads (especially AST) are slow.
- `actions/` — offline pipeline scripts for the context model, run in order:
  `prepare_catmeows.py` → `extract_ast_embeddings.py` →
  `extract_prosody.py` (optional) → `train_ast_head.py`.
  (`train_context.py` is an optional raw-waveform baseline, ~36%; `train.py`
  is the original upstream M5 training script driven by `configs/config.json`.)
- Prosodic features (`models/prosody.py`, librosa pyin) can be concatenated
  onto the AST embedding, but are **off by default** — the head bundle carries
  a `use_prosody` flag and `train_ast_head.py` only enables it if it beats
  AST-only on the unseen-cat test set (it didn't; see `EXPERIMENTS.md`). The
  app's inference wrapper honours that flag, so it skips pyin unless a future
  head turns it on.

## Data

- **Do not run `dvc pull`** — the upstream gdrive remote is private and returns
  401. The `.dvc` files are vestigial.
- Dataset sources: CatSound → Zenodo record 4724180 (public page has a 2.5 MB
  sample only; the full 12.2 GB lives behind the authors' Google Drive — the
  same data the dead DVC remote pointed at); CatMeows → HF
  `oliveirabruno01/openfarm-catmeows` (fetched by `actions/prepare_catmeows.py`),
  with HF `zeddez/CatMeows` as a Parquet backup mirror if that one breaks.
- `data/catmeows_context/` and `data/ast_embeddings/` are gitignored generated
  artifacts; regenerate them with the `actions/` scripts rather than looking
  for them in git history.

## Conventions

- All torch loading uses `map_location="cpu"`; keep new code CPU-safe.
- Sample rates are model-specific constants (`SENTIMENT_SAMPLE_RATE = 44100`,
  AST at 16000) — resample explicitly, never assume the input rate.
- Class label maps and emoji strings live at the top of `app.py`; if classes
  change, update `SENTIMENT_CLASSES`/`CONTEXT_EMOJIS` together with the model.
- Git identity for commits is repo-local (`lilllee <kopqwe147@gmail.com>`);
  don't change it to the global identity.
- After any training/evaluation run, log the result in `EXPERIMENTS.md`
  (date, data, setup, accuracy on unseen cats — not cross-validation).

## Phone access gotchas

- Mobile browsers allow microphone recording **only over HTTPS**; plain
  `http://<pc-ip>:8501` works for upload but not in-browser recording.
- HTTPS is provided by `tailscale serve --bg 8501`. The serve config survives
  reboots, but Streamlit does not — after a reboot, restart the app.
- Curling the `*.ts.net` URL from this machine fails **by design**; test from
  another tailnet device (the phone). Personal URLs live in `CLAUDE.local.md`.
