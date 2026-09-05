# Wake Word Trainer

Samostatná webová aplikace pro nahrávání hlasových vzorků a trénování vlastního wake word modelu
pomocí [microWakeWord](https://github.com/kahrendt/microWakeWord) (TensorFlow → kvantizovaný streamovaný
TensorFlow Lite model použitelný v ESPHome `micro_wake_word`).

- **Frontend:** Vite + React + TypeScript + Material UI (světlé/tmavé téma podle systému)
- **Backend:** FastAPI (Python 3.11), trénink běží jako subprocess, průběh se streamuje přes SSE
- **Audio:** Web Audio API (AudioWorklet) → WAV 16 kHz / mono / 16-bit PCM; na serveru normalizace přes FFmpeg

## Spuštění

```bash
# CPU (funguje všude)
docker compose up --build

# NVIDIA GPU (Docker + nvidia-container-toolkit, funguje i ve WSL2)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# …nebo: cp .env.example .env  (obsahuje COMPOSE_FILE=…gpu.yml) a pak stačí `docker compose up --build`
```

Aplikace běží na <http://localhost:8000>. Všechna data (nahrávky, datasety, modely) jsou ve svazku `./data`.

> Mikrofon v prohlížeči funguje jen v „secure context“ – tedy na `localhost` nebo přes HTTPS.
> Pokud k aplikaci přistupujete z jiného počítače v síti, použijte HTTPS reverse proxy nebo SSH tunel
> (`ssh -L 8000:localhost:8000 …`).

Volitelně lze nastavit HTTP Basic auth proměnnými `APP_USER` / `APP_PASSWORD` (viz `.env.example`).

## Postup

1. **Konfigurace** – zadejte wake word (např. `chaloupko`), délku vzorku a případně upravte parametry trénování
   (kroky, learning rate, batch size, počet augmentací…).
2. **Nahrávání** – tlačítkem nebo mezerníkem nahrajete vzorek pevné délky. K dispozici je sériové nahrávání
   s odpočtem, přehrání po nahrání, přehrát vše, výběr mikrofonu, kontrola kvality (oříznutí, přebuzení, ticho),
   mini vlnovka u každé nahrávky, hromadné mazání s možností vrátit zpět a import existujících audio souborů
   (drag & drop). Doporučeno 20–40 vzorků. Volitelně můžete nahrát i vlastní negativní vzorky (jiná řeč, hluk).
3. **Negativní datasety** – základní dataset (Google *mini_speech_commands*, ~180 MB) se stáhne automaticky
   před prvním tréninkem. Volitelně lze stáhnout předpočítané spektrogramy `dinner_party` / `dinner_party_eval`
   z microWakeWord (Hugging Face).
4. **Trénování** – spustí pipeline: augmentace pozitivních vzorků → spektrogramy (microWakeWord micro-frontend)
   → syntetický šum + ambientní záznam pro odhad falešných aktivací → trénink MixedNet → kvantizace a konverze
   na streamovaný `.tflite`. Průběh (kroky, loss, přesnost, validace, log) se zobrazuje živě.
5. **Stažení** – po dokončení stáhnete `<wakeword>.tflite` a manifest JSON pro ESPHome.

## Použití v ESPHome

```yaml
micro_wake_word:
  models:
    - model: chaloupko.json   # manifest vedle .tflite, oba soubory v config složce ESPHome
```

Hodnotu `probability_cutoff` v manifestu (výchozí 0.97) podle potřeby snižte (snazší spouštění) nebo zvyšte
(méně falešných aktivací).

## API

| Metoda | Cesta | Popis |
| --- | --- | --- |
| GET/PUT | `/api/config` | Konfigurace projektu (wake word, parametry trénování) |
| GET | `/api/recordings?kind=positive|negative` | Seznam nahrávek vč. vlnovky a kontroly kvality |
| POST | `/api/recordings` | Upload (multipart `file`, `kind`) – uloží do `/data/positive_samples` nebo `/data/negative_samples` |
| GET | `/api/recordings/{kind}/{id}` | Přehrání WAV |
| DELETE | `/api/recordings/{kind}/{id}` | Smazání (do koše), `POST …/restore` vrátí zpět |
| GET | `/api/datasets` · POST `/api/datasets/{id}/download` | Negativní datasety |
| POST | `/api/train` · POST `/api/train/cancel` · GET `/api/train` | Spuštění / zrušení / snapshot trénování |
| GET | `/api/train/status` | SSE stream (`snapshot`, `state`, `log`) |
| GET | `/api/train/model` | Stažení posledního natrénovaného `.tflite` |
| GET | `/api/jobs` · `/api/jobs/{id}/model` · `/api/jobs/{id}/manifest` · `/api/jobs/{id}/log` | Historie běhů |

## Struktura

```
backend/app        FastAPI (config, recordings, datasets, training/SSE, static frontend)
backend/trainer    Trénovací pipeline (run.py) + audio pomocné funkce
frontend           Vite + React + MUI
data/              (vytvoří se) nahrávky, datasety, cache spektrogramů, běhy trénování
```

## Vývoj bez Dockeru

```bash
# backend
cd backend && pip install -r requirements.txt tensorflow==2.19.0
DATA_DIR=../data uvicorn app.main:app --reload
# frontend (proxy /api -> :8000)
cd frontend && npm install && npm run dev
```
