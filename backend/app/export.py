"""ESPHome / Wyoming snippets and the per-run ZIP export."""
from __future__ import annotations

import io
import zipfile

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .jobs import _job_file, _read_json

router = APIRouter(prefix="/api", tags=["export"])


def wyoming_readme(slug: str, wake_word: str) -> str:
    return f"""openWakeWord model '{wake_word}' for Wyoming satellites / the Home Assistant openWakeWord add-on.

Home Assistant add-on:  copy {slug}.tflite into /share/openwakeword/ (Samba or SSH add-on), restart the add-on,
                        then pick "{slug}" as the wake word in the Assist pipeline / satellite.

wyoming-openwakeword (Docker):
  services:
    openwakeword:
      image: rhasspy/wyoming-openwakeword
      command: --preload-model {slug} --custom-model-dir /custom --threshold 0.5
      volumes:
        - ./models:/custom          # put {slug}.tflite here
      ports:
        - "10400:10400"

wyoming-satellite: add `--wake-uri tcp://<host>:10400 --wake-word-name {slug}`.
Threshold: 0.5 is the openWakeWord default; raise it on false activations, lower it if the word is hard to trigger
(the "auto threshold" in the training summary is a good starting point).
"""


def esphome_snippet(slug: str, wake_word: str) -> str:
    return f"""# Example ESPHome configuration for the "{wake_word}" wake word.
# Copy {slug}.tflite and {slug}.json next to this YAML (or point `model:` to a URL).
micro_wake_word:
  models:
    - model: {slug}.json
  on_wake_word_detected:
    - logger.log:
        format: "Wake word detected: %s"
        args: ['x.c_str()']
    # - voice_assistant.start:
    #     wake_word: !lambda return x;

# Tuning: if "{wake_word}" is hard to trigger, lower "probability_cutoff" in {slug}.json;
# if it triggers falsely, raise it (0.5 - 0.99).
"""


@router.get("/jobs/{job_id}/export")
def job_export(job_id: str):
    model, job = _job_file(job_id, ".tflite")
    job_dir = model.parent
    slug = job.get("slug", "wakeword")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model, f"{slug}/{model.name}")
        manifest = job_dir / f"{slug}.json"
        if manifest.exists():
            zf.write(manifest, f"{slug}/{manifest.name}")
        if (job.get("training") or {}).get("target") == "wyoming":
            zf.writestr(f"{slug}/WYOMING.txt", wyoming_readme(slug, job.get("wake_word", slug)))
        else:
            zf.writestr(f"{slug}/esphome-example.yaml", esphome_snippet(slug, job.get("wake_word", slug)))
        for extra in ("training_parameters.yaml", "result.json", "train.log"):
            if (job_dir / extra).exists():
                zf.write(job_dir / extra, f"{slug}/training/{extra}")
        readme = (
            f"Wake word model '{job.get('wake_word')}' trained with Wake Word Trainer / microWakeWord on {job.get('created_at')}.\n\n"
            f"Files:\n  {slug}.tflite         quantised streaming TensorFlow Lite model\n  {slug}.json           ESPHome micro_wake_word manifest\n"
            f"  esphome-example.yaml  example ESPHome configuration\n  training/             training parameters, metrics and log\n"
        )
        zf.writestr(f"{slug}/README.txt", readme)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{slug}-wakeword.zip"'})


