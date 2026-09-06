# Wake Word Trainer

> 🇬🇧 English version: [README.md](README.md)

Samostatná webová aplikace pro nahrávání hlasových vzorků, trénování vlastního **wake word** modelu pomocí
[microWakeWord](https://github.com/kahrendt/microWakeWord) (TensorFlow → kvantizovaný streamovaný TensorFlow Lite)
a otestování výsledku přímo v prohlížeči. Výstupní `.tflite` + manifest fungují přímo s ESPHome komponentou
[`micro_wake_word`](https://esphome.io/components/micro_wake_word).

![Wake Word Trainer – tmavé téma](docs/screenshots/hero-dark.png)

## Funkce

- **Nahrávání v prohlížeči** – Web Audio API (AudioWorklet), vzorky pevné délky, WAV 16 kHz / mono / 16-bit PCM.
  Sériové nahrávání s odpočtem, přehrání po nahrání, přehrát vše, výběr mikrofonu, kontrola kvality (oříznutý začátek/konec,
  přebuzení, příliš tiché, ticho), mini vlnovky, hromadné mazání s vrácením zpět, import přetažením, klávesové zkratky
  (mezerník / R / Esc).
- **Negativní data vyřešená za vás** – Google *mini_speech_commands* se stáhne automaticky; generuje se syntetický šum a dlouhý
  „ambientní“ záznam pro odhad falešných aktivací za hodinu; volitelně sady `dinner_party` z microWakeWord.
- **Trénovací pipeline** – ořez ticha + augmentace (audiomentations) → spektrogramy micro-frontendu → trénink MixedNet
  originálním `microwakeword.model_train_eval` → int8 kvantizovaný streamovaný `.tflite`. Živý průběh přes SSE: kroky, loss,
  přesnost, validační tabulka, log.
- **Živý test** – audio z mikrofonu se streamuje WebSocketem na server, kde běží tentýž kvantizovaný streamovaný model se
  stejným micro-frontendem a klouzavým průměrem jako v ESPHome. Vyhodnocení modelu na všech uložených nahrávkách ukáže,
  které vzorky model nerozpozná a které negativní nahrávky ho spustí.
- **Manifest pro ESPHome** – `probability_cutoff` se odvodí z ROC křivky na testovací sadě; doladíte ho v testovací kartě.
- **UI** – React + Material UI, světlé/tmavé téma podle systému, čeština/angličtina podle jazyka prohlížeče (přepínač v hlavičce).
- **Docker** – spuštění jedním příkazem, volitelný build pro NVIDIA GPU (funguje ve WSL2), volitelná HTTP Basic autentizace.

| Nahrávání | Trénování |
| --- | --- |
| ![Nahrávání](docs/screenshots/recording.png) | ![Trénování](docs/screenshots/training-progress.png) |

| Živý test a vyhodnocení | Datasety |
| --- | --- |
| ![Test](docs/screenshots/test.png) | ![Datasety](docs/screenshots/datasets.png) |

## Spuštění

```bash
# CPU (funguje všude)
docker compose up --build

# NVIDIA GPU (Docker + nvidia-container-toolkit; funguje i ve WSL2)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# …nebo: cp .env.example .env   (nastaví COMPOSE_FILE s GPU variantou) a pak stačí `docker compose up --build`
```

Aplikace běží na <http://localhost:8000>. Všechna data (nahrávky, datasety, cache, modely) jsou ve svazku `./data`
(kontejner běží jako root, soubory v `./data` proto patří rootovi).

> Mikrofon v prohlížeči funguje jen v „secure context“ – `localhost` nebo HTTPS. Z jiného počítače použijte SSH tunel
> (`ssh -L 8000:localhost:8000 host`) nebo HTTPS reverse proxy.

**GPU není nutné.** Model je malý (~30 k parametrů), výchozích 4 000 kroků trvá na CPU jednotky až nižší desítky minut.
První běh navíc jednou předpočítá spektrogramy negativního datasetu (~1–2 min, cache v `data/features_cache`).

Volitelná Basic autentizace: proměnné `APP_USER` / `APP_PASSWORD` (viz `.env.example`).

## Postup

1. **Konfigurace** – wake word (např. `chaloupko`), délka vzorku, případně parametry trénování.
2. **Nahrávání** – doporučeno 20–40 vzorků, ideálně více mluvčích, vzdáleností a intonací. Volitelně negativní vzorky.
3. **Datasety** – základní negativní dataset se stáhne automaticky před prvním tréninkem.
4. **Trénování** – živý průběh; nejlepší váhy (podle *average viable recall*) se zkonvertují a kvantizují.
5. **Stažení** – `<wakeword>.tflite` a `<wakeword>.json` (manifest pro ESPHome).
6. **Test** – živý poslech, ladění prahu/okna, vyhodnocení na nahrávkách.

### Použití v ESPHome

```yaml
micro_wake_word:
  models:
    - model: chaloupko.json   # manifest vedle chaloupko.tflite ve složce konfigurace ESPHome
```

Když se slovo spouští těžko, `probability_cutoff` snižte; při falešných aktivacích ho zvyšte.

## API

Popis endpointů je v anglickém [README.md](README.md#api).

## Struktura

```
backend/app        FastAPI (konfigurace, nahrávky, datasety, trénink + SSE, živý test, statický frontend)
backend/trainer    Trénovací pipeline (run.py) a audio pomocné funkce
frontend           Vite + React + TypeScript + Material UI
docs/screenshots   Obrázky pro README
data/              (vytvoří se za běhu) nahrávky, datasety, cache spektrogramů, běhy trénování
```

## Poděkování

Postaveno na [microWakeWord](https://github.com/kahrendt/microWakeWord) Kevina Ahrendta (Apache 2.0). Negativní řeč:
Google Speech Commands (CC BY 4.0). Natrénované modely přebírají licence použitých dat.
