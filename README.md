# Wake Word Trainer


A self-contained web app for recording voice samples, training your own **wake word** model with
[microWakeWord](https://github.com/kahrendt/microWakeWord) (TensorFlow → quantised streaming TensorFlow Lite),
and testing the result live in the browser. The output `.tflite` + manifest works directly with the ESPHome
[`micro_wake_word`](https://esphome.io/components/micro_wake_word) component.

![Wake Word Trainer – dark theme](docs/screenshots/hero-dark.png)

## Features

- **Sample recording in the browser** – Web Audio API (AudioWorklet), fixed-length clips, WAV 16 kHz / mono / 16-bit PCM.
  Series recording with countdown, playback after recording, play-all, microphone selection, quality checks
  (clipped start/end, clipping, too quiet, silence), mini waveforms, bulk delete with undo, drag & drop import,
  keyboard shortcuts (Space / R / Esc).
- **Negative data handled for you** – Google *mini_speech_commands* is downloaded automatically; synthetic noise and a long
  "ambient" recording for false-accept-per-hour estimation are generated; optional microWakeWord `dinner_party` sets.
- **Training pipeline** – silence trimming + augmentation (audiomentations) → micro-frontend spectrograms → MixedNet training with the
  original `microwakeword.model_train_eval` → int8 quantised streaming `.tflite`. Live progress over SSE: steps, loss,
  accuracy, validation table, log.
- **Live test** – microphone audio is streamed over WebSocket to the server, which runs the very same quantised streaming
  model with the same micro-frontend and sliding-window average as ESPHome. Evaluate the model on all stored recordings
  to see which samples are missed and which negatives trigger it.
- **ESPHome manifest** – `probability_cutoff` is derived from the test-set ROC curve; tune it in the test card and copy it over.
- **Multiple projects** – each wake word lives in its own project (recordings, runs, settings); switch in the header.
- **Shared recording link** – generate a link (`/contribute?token=…`) so family or colleagues can record samples for a
  project from their own device without logging in; they only see their own recordings.
- **Hard negatives** – word fragments and swapped halves of your recordings are used as extra negatives so similar words
  do not trigger the model (optional).
- **Charts** – validation loss and recall/accuracy over steps, test-set ROC curve.
- **Robust runs** – an interrupted run (container restart) is detected and can be resumed from its checkpoint; old runs
  are pruned automatically (`KEEP_JOBS`), heavy intermediate files are removed after a successful run.
- **Export** – ZIP bundle with the `.tflite`, the ESPHome manifest, an example ESPHome YAML and the training log.
- **UI** – React + Material UI, light/dark theme following the system, Czech/English following the browser language (manual override in the header).
- **Docker** – one command to run, optional NVIDIA GPU build (works in WSL2), runs as an unprivileged user (`PUID`/`PGID`),
  optional HTTP Basic auth. CI runs unit tests, the frontend build and a Playwright smoke test of the Docker image.

| Recording | Training |
| --- | --- |
| ![Recording](docs/screenshots/recording.png) | ![Training](docs/screenshots/training-progress.png) |

| Live test & evaluation | Datasets |
| --- | --- |
| ![Test](docs/screenshots/test.png) | ![Datasets](docs/screenshots/datasets.png) |

## Quick start

```bash
# CPU (works everywhere)
docker compose up --build

# NVIDIA GPU (Docker + nvidia-container-toolkit; works in WSL2)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# …or: cp .env.example .env   (sets COMPOSE_FILE to include the GPU override) and just `docker compose up --build`
```

Open <http://localhost:8000>. Everything (recordings, datasets, feature cache, models) lives in `./data`, owned by
`PUID`/`PGID` (default 1000:1000 – set them in `.env` to your `id -u` / `id -g`).

> The microphone only works in a secure context – `localhost` or HTTPS. From another machine use an SSH tunnel
> (`ssh -L 8000:localhost:8000 host`) or an HTTPS reverse proxy.

**GPU is optional.** The model is tiny (~30 k parameters); the default 4,000 steps take minutes on a CPU. The first run
additionally pre-computes spectrograms of the negative dataset once (~1–2 min, cached in `data/features_cache`).

Optional Basic auth: set `APP_USER` / `APP_PASSWORD` (see `.env.example`).

## Workflow

1. **Configure** – wake word (e.g. `chaloupko`), sample length, optionally training parameters (steps, learning rate, batch
   size, augmentations per sample, model window, negative class weight).
2. **Record** – 20–40 samples recommended, ideally several speakers, distances and intonations. Optionally record negative
   samples (other speech, similar words, room noise).
3. **Datasets** – the base negative dataset downloads automatically before the first run; extra sets are optional.
4. **Train** – watch the progress live; the best weights (by *average viable recall*) are converted and quantised.
5. **Download** – `<wakeword>.tflite` and `<wakeword>.json` (ESPHome manifest).
6. **Test** – listen live, tune threshold/window, evaluate on your recordings.

Need more voices? Open **Shared recording link** in the configuration card and send the link around. Contributors
need HTTPS (or localhost) for the microphone – put an HTTPS reverse proxy (e.g. Caddy) in front when sharing on a LAN.

### Using the model in ESPHome

```yaml
micro_wake_word:
  models:
    - model: chaloupko.json   # manifest next to chaloupko.tflite in the ESPHome config folder
```

Lower `probability_cutoff` if the word is hard to trigger; raise it on false activations.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET / PUT | `/api/config` | Current project configuration (wake word, training parameters) |
| GET / POST / DELETE | `/api/projects` · `/api/projects/{id}/select` · `/api/projects/{id}/share` | Projects and sharing links |
| GET / POST | `/api/contribute/info` · `/api/contribute/recordings` (`?token=…&name=…`) | Contributor (record-only) access |
| GET | `/api/recordings?kind=positive\|negative` | Recordings incl. waveform peaks and quality analysis |
| POST | `/api/recordings` | Upload (multipart `file`, `kind`) → `/data/positive_samples` or `/data/negative_samples` |
| GET | `/api/recordings/{kind}/{id}` | Play a WAV |
| DELETE | `/api/recordings/{kind}/{id}` | Soft delete (trash), `POST …/restore` restores |
| GET / POST | `/api/datasets` · `/api/datasets/{id}/download` | Negative datasets |
| POST / GET | `/api/train` · `/api/train/resume` · `/api/train/cancel` · `/api/train` | Start / resume / cancel / snapshot |
| GET | `/api/train/status` | SSE stream (`snapshot`, `state`, `log`) |
| GET | `/api/train/model` | Latest trained `.tflite` |
| GET | `/api/jobs` · `/api/jobs/{id}/model` · `/api/jobs/{id}/manifest` · `/api/jobs/{id}/export` · `/api/jobs/{id}/log` | Run history, ZIP export |
| WS | `/api/test/ws?job_id=…&cutoff=…&window=…` | Live test: binary int16 16 kHz PCM in → JSON probabilities/detections out |
| POST | `/api/jobs/{id}/evaluate?cutoff=…&window=…` | Evaluate a model on the stored recordings |

## Project layout

```
backend/app        FastAPI (config, recordings, datasets, training + SSE, live test, static frontend)
backend/trainer    Training pipeline (run.py) and audio helpers
backend/tests      pytest suite (no TensorFlow needed: `pip install -r backend/requirements-dev.txt && cd backend && pytest`)
frontend           Vite + React + TypeScript + Material UI
tests/e2e          Playwright smoke test used in CI against the Docker image
docs/screenshots   README images
data/              (created at runtime) negative_datasets/, features_cache/, projects/<id>/{positive_samples,negative_samples,jobs}
```

## Development without Docker

```bash
cd backend && pip install -r requirements.txt tensorflow==2.19.0
git clone https://github.com/kahrendt/microWakeWord /opt/microWakeWord && pip install --no-deps -e /opt/microWakeWord
DATA_DIR=../data uvicorn app.main:app --reload
# frontend (proxies /api to :8000)
cd frontend && npm install && npm run dev
```

## Credits

Built on [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt (Apache 2.0). Negative speech data:
Google Speech Commands (CC BY 4.0). Trained models inherit the licences of the data used.
