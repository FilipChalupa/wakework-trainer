"""End-to-end training pipeline executed as a subprocess by the API.

Progress is reported on stdout as lines prefixed with ``@@`` followed by JSON; everything else
is plain log output (microWakeWord/TensorFlow logs are parsed by the API for step progress).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import runpy
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from trainer.audio_data import (
    SR,
    WavClips,
    build_ambient_clip,
    ensure_synthetic_noise,
    stable_split,
)

FEATURE_SCALE = 0.0390625  # microWakeWord stores uint16 spectrograms; float = uint16 * 0.0390625
SPEECH_CACHE_VERSION = "v1"


def emit(event: str, **data) -> None:
    print("@@" + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def log(message: str) -> None:
    emit("log", message=message)


def stage(key: str, name: str, message_key: str | None = None, message: str | None = None, params: dict | None = None, **extra) -> None:
    emit("stage", key=key, name=name, message_key=message_key, message=message, params=params or {}, **extra)


# --------------------------------------------------------------------------------------
class FeatureWriter:
    """Writes spectrograms produced by microWakeWord's SpectrogramGeneration into RaggedMmap."""

    def __init__(self) -> None:
        from microwakeword.audio.audio_utils import generate_features_for_clip

        probe = generate_features_for_clip((np.random.default_rng(0).standard_normal(SR) * 8000).astype(np.int16), 10)
        probe_max = float(probe.max()) if probe.size else 0.0
        # Detect whether the C frontend returns already scaled floats (0..~26) or raw uint16 values.
        self.to_uint16 = probe_max <= 64.0
        log(f"Feature frontend probe: shape={probe.shape} max={probe_max:.2f} -> {'scaled float' if self.to_uint16 else 'raw uint16'} domain")

    def convert(self, spectrogram: np.ndarray) -> np.ndarray:
        if spectrogram.dtype == np.uint16:
            return spectrogram
        if self.to_uint16:
            return np.clip(np.rint(spectrogram / FEATURE_SCALE), 0, 65535).astype(np.uint16)
        return np.clip(np.rint(spectrogram), 0, 65535).astype(np.uint16)

    def write(self, out_dir: Path, generator, expected: int | None, label: str, batch_size: int = 100) -> int:
        from mmap_ninja.ragged import RaggedMmap

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        last = time.time()

        def wrapped():
            nonlocal count, last
            for spec in generator:
                if spec.shape[0] < 4:
                    continue
                count += 1
                if expected and (count % 25 == 0 or time.time() - last > 1.0):
                    last = time.time()
                    emit("progress", current=count, total=expected)
                yield self.convert(np.asarray(spec))

        RaggedMmap.from_generator(out_dir=str(out_dir), sample_generator=wrapped(), batch_size=batch_size, verbose=False)
        log(f"{label}: {count} spectrograms -> {out_dir}")
        return count


def make_augmenter(duration_s: float, background: list[str], positive: bool, seed: int | None = None):
    from microwakeword.audio.augmentation import Augmentation

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    probabilities = {
        "SevenBandParametricEQ": 0.2,
        "TanhDistortion": 0.1,
        "PitchShift": 0.25 if positive else 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.3,
        "AddBackgroundNoise": 0.6 if background else 0.0,
        "Gain": 1.0,
        "GainTransition": 0.2,
        "RIR": 0.0,
    }
    return Augmentation(
        augmentation_duration_s=duration_s,
        augmentation_probabilities=probabilities,
        background_paths=background,
        background_min_snr_db=-3 if positive else -10,
        background_max_snr_db=15,
        min_gain_db=-30,
        max_gain_db=0,
        min_jitter_s=0.15 if positive else 0.0,
        max_jitter_s=0.25 if positive else 0.8,
        truncate_randomly=not positive,
    )


def spectrogram_generator(clips, augmenter, split: str, repeat: int, slide_frames: int | None):
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    gen = SpectrogramGeneration(clips=clips, augmenter=augmenter, step_ms=10, slide_frames=slide_frames)
    return gen.spectrogram_generator(split=split, repeat=repeat)


# --------------------------------------------------------------------------------------
def prepare_positives(job: dict, writer: FeatureWriter, features_dir: Path, background: list[str]) -> tuple[int, int]:
    """Returns (effective clip_duration_ms, number of training clips)."""
    training = job["training"]
    positive_paths = sorted(Path(job["positive_dir"]).glob("*.wav"))
    if len(positive_paths) < 3:
        raise RuntimeError("At least 3 positive samples are required")
    splits = stable_split(positive_paths)
    clips = WavClips(splits, trim=True)
    durations = clips.durations("train") + clips.durations("validation") + clips.durations("test")
    p90 = float(np.percentile(durations, 90))
    clip_ms = int(training["clip_duration_ms"])
    needed_ms = int(np.ceil((p90 + 0.35) * 10) * 100)
    if needed_ms > clip_ms:
        log(f"Wake word samples are long (p90 {p90:.2f}s); raising clip_duration_ms {clip_ms} -> {needed_ms}")
        clip_ms = needed_ms
    log(f"Positive samples: {len(positive_paths)} (train {len(splits['train'])}, val {len(splits['validation'])}, test {len(splits['test'])}); trimmed duration median {np.median(durations):.2f}s, p90 {p90:.2f}s")

    reps = max(1, int(training["augmentations_per_sample"]))
    duration_s = clip_ms / 1000.0 + 0.2
    plan = [
        ("training", "train", reps, 5),
        ("validation", "validation", max(2, reps // 4), 3),
        ("testing", "test", max(2, reps // 4), 1),
    ]
    for set_name, split, repeat, slide in plan:
        expected = len(splits[split]) * repeat * slide
        stage("preparing", "Preparing data", "augment_positive", f"Augmenting positive samples ({set_name})", {"set": set_name}, current=0, total=expected)
        augmenter = make_augmenter(duration_s, background, positive=True)
        writer.write(
            features_dir / "positive" / set_name / "wakeword_mmap",
            spectrogram_generator(clips, augmenter, split, repeat, slide),
            expected,
            f"positive/{set_name}",
        )
    return clip_ms, len(splits["train"])


def list_speech_commands(job: dict) -> list[Path]:
    dataset_dir = Path(job["datasets_dir"]) / "mini_speech_commands"
    return sorted(
        p
        for p in dataset_dir.rglob("*.wav")
        if "_background_noise_" not in p.parts and "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def prepare_speech_commands(job: dict, writer: FeatureWriter) -> Path | None:
    wavs = list_speech_commands(job)
    if not wavs:
        log("mini_speech_commands not found - training without generic speech negatives (model quality will suffer)")
        return None
    cache_dir = Path(job["feature_cache_dir"]) / f"mini_speech_commands_{SPEECH_CACHE_VERSION}"
    marker = cache_dir / "READY"
    if marker.exists():
        log(f"Using cached speech negative features from {cache_dir}")
        return cache_dir
    splits = stable_split(wavs, validation=0.1, test=0.1)
    clips = WavClips(splits, trim=False, cache=False)
    plan = [("training", "train", 1), ("validation", "validation", 1), ("testing", "test", 1)]
    for set_name, split, repeat in plan:
        expected = len(splits[split]) * repeat
        stage("preparing", "Preparing data", "speech_features", f"Spectrograms of negative speech ({set_name}, {expected} clips)", {"set": set_name, "count": expected}, current=0, total=expected)
        augmenter = make_augmenter(2.0, [], positive=False, seed=42)
        writer.write(cache_dir / set_name / "speech_mmap", spectrogram_generator(clips, augmenter, split, repeat, None), expected, f"speech/{set_name}")
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return cache_dir


def prepare_noise_and_user_negatives(job: dict, writer: FeatureWriter, features_dir: Path, noise_files: list[Path], clip_ms: int) -> Path:
    user_negatives = sorted(Path(job["negative_dir"]).glob("*.wav"))
    out_dir = features_dir / "noise"
    duration_s = clip_ms / 1000.0 + 0.2
    noise_reps = 12
    user_reps = 15
    plan = [("training", 1.0), ("validation", 0.25), ("testing", 0.25)]
    for set_name, fraction in plan:
        n_reps = max(1, int(noise_reps * fraction))
        u_reps = max(1, int(user_reps * fraction))
        expected = len(noise_files) * n_reps + len(user_negatives) * u_reps
        stage("preparing", "Preparing data", "noise_features", f"Noise and custom negative recordings ({set_name})", {"set": set_name}, current=0, total=expected)

        def gen():
            noise_clips = WavClips({"train": noise_files, "validation": noise_files, "test": noise_files}, trim=False, normalize=False)
            yield from spectrogram_generator(noise_clips, make_augmenter(duration_s, [], positive=False), "train", n_reps, None)
            if user_negatives:
                user_clips = WavClips({"train": user_negatives, "validation": user_negatives, "test": user_negatives}, trim=False)
                yield from spectrogram_generator(user_clips, make_augmenter(duration_s, [str(noise_files[0].parent)], positive=False), "train", u_reps, None)

        writer.write(out_dir / set_name / "noise_mmap", gen(), expected, f"noise/{set_name}")
    return out_dir


def prepare_ambient(job: dict, writer: FeatureWriter, features_dir: Path, noise_files: list[Path], minutes: float = 4.0) -> Path:
    from microwakeword.audio.audio_utils import generate_features_for_clip

    speech = list_speech_commands(job)
    user_negatives = sorted(Path(job["negative_dir"]).glob("*.wav"))
    out_dir = features_dir / "ambient"
    for set_name, seed in (("validation_ambient", 7), ("testing_ambient", 11)):
        rng = random.Random(seed)
        splits = stable_split(speech) if speech else {"validation": [], "test": []}
        pool = list(splits["validation" if set_name.startswith("validation") else "test"]) + user_negatives * 5
        stage("preparing", "Preparing data", "ambient_features", f"Long ambient recording for false-accept measurement ({set_name})", {"set": set_name}, current=0, total=2)
        if not pool:
            pool = noise_files

        def gen():
            for i in range(2):
                clip = build_ambient_clip(pool, minutes * 60 / 2, rng, noise_files)
                yield generate_features_for_clip(clip, 10)

        writer.write(out_dir / set_name / "ambient_mmap", gen(), 2, f"ambient/{set_name}")
    return out_dir


# --------------------------------------------------------------------------------------
def build_training_config(job: dict, job_dir: Path, feature_sets: list[dict], clip_ms: int) -> Path:
    training = job["training"]
    steps = int(training["training_steps"])
    eval_interval = max(25, min(int(training["eval_step_interval"]), steps))
    config = {
        "window_step_ms": 10,
        "train_dir": str(job_dir / "trained_model"),
        "features": feature_sets,
        "training_steps": [steps],
        "positive_class_weight": [float(training["positive_class_weight"])],
        "negative_class_weight": [float(training["negative_class_weight"])],
        "learning_rates": [float(training["learning_rate"])],
        "batch_size": int(training["batch_size"]),
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": eval_interval,
        "clip_duration_ms": int(clip_ms),
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    path = job_dir / "training_parameters.yaml"
    path.write_text(yaml.dump(config, default_flow_style=False))
    emit("training_config", total_steps=steps, eval_step_interval=eval_interval, clip_duration_ms=clip_ms)
    return path


def run_microwakeword(config_path: Path) -> None:
    argv = [
        "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]
    log("Running: python -m " + " ".join(argv))
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)
    except SystemExit as exc:  # argparse / absl may sys.exit(0)
        if exc.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv


def parse_roc(text: str) -> tuple[float | None, list[dict]]:
    """Parses microWakeWord's tflite_streaming_roc.txt (AUC + points on the FAPH/FRR curve)."""
    auc = None
    points = []
    for line in text.splitlines():
        m = re.match(r"AUC ([\d.]+)", line)
        if m:
            auc = float(m.group(1))
            continue
        m = re.match(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)", line)
        if m:
            points.append({"cutoff": float(m.group(1)), "frr": float(m.group(2)), "faph": float(m.group(3))})
    return auc, points


def summarize_roc(auc: float | None, points: list[dict]) -> tuple[dict | None, float]:
    """Returns (structured summary, suggested probability cutoff for the ESPHome manifest)."""
    cutoff = 0.97
    if not points:
        return None, cutoff
    clean = [p for p in points if p["faph"] <= 0.5] or [min(points, key=lambda p: p["faph"])]
    best = min(clean, key=lambda p: (p["frr"], -p["cutoff"]))
    cutoff = float(min(0.97, max(0.6, round(best["cutoff"] + 0.05, 2))))
    summary = {"auc": auc, "cutoff": best["cutoff"], "frr": best["frr"], "faph": best["faph"], "manifest_cutoff": cutoff}
    return summary, cutoff


def write_manifest(job: dict, job_dir: Path, clip_ms: int, model_name: str, probability_cutoff: float) -> None:
    manifest = {
        "type": "micro",
        "wake_word": job["wake_word"],
        "author": "wakeword-trainer",
        "website": "https://github.com/kahrendt/microWakeWord",
        "model": model_name,
        "trained_languages": ["cs"],
        "version": 2,
        "micro": {
            "probability_cutoff": probability_cutoff,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
    (job_dir / f"{job['slug']}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    job_dir = Path(job["job_dir"])
    features_dir = job_dir / "features"
    started = time.time()

    stage("preparing", "Preparing data", "init_tf", "Initialising TensorFlow and microWakeWord", status="preparing")
    import tensorflow as tf  # noqa: F401  (import early so failures show up immediately)

    gpus = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow {tf.__version__}; GPU devices: {[g.name for g in gpus] or 'none (CPU training)'}")

    writer = FeatureWriter()
    noise_dir = Path(job["feature_cache_dir"]) / "synthetic_noise"
    noise_files = ensure_synthetic_noise(noise_dir)
    background_dirs = [str(noise_dir)]
    if any(Path(job["negative_dir"]).glob("*.wav")):
        background_dirs.append(job["negative_dir"])

    clip_ms, n_train = prepare_positives(job, writer, features_dir, background_dirs)
    speech_dir = prepare_speech_commands(job, writer)
    noise_feat_dir = prepare_noise_and_user_negatives(job, writer, features_dir, noise_files, clip_ms)
    ambient_dir = prepare_ambient(job, writer, features_dir, noise_files)

    feature_sets = [
        {"features_dir": str(features_dir / "positive"), "sampling_weight": 3.0, "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
        {"features_dir": str(noise_feat_dir), "sampling_weight": 3.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": str(ambient_dir), "sampling_weight": 0.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "split", "type": "mmap"},
    ]
    if speech_dir is not None:
        feature_sets.append({"features_dir": str(speech_dir), "sampling_weight": 8.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"})
    datasets_dir = Path(job["datasets_dir"])
    if (datasets_dir / "dinner_party").is_dir() and any((datasets_dir / "dinner_party").rglob("*_mmap")):
        feature_sets.append({"features_dir": str(datasets_dir / "dinner_party"), "sampling_weight": 6.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"})
        log("Using extra negative dataset: dinner_party")
    if (datasets_dir / "dinner_party_eval").is_dir() and any((datasets_dir / "dinner_party_eval").rglob("*_mmap")):
        feature_sets.append({"features_dir": str(datasets_dir / "dinner_party_eval"), "sampling_weight": 0.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "split", "type": "mmap"})
        log("Using extra ambient evaluation dataset: dinner_party_eval")

    config_path = build_training_config(job, job_dir, feature_sets, clip_ms)
    log(f"Data preparation finished in {time.time() - started:.0f}s")

    stage("training", "Training", "train_steps", f"Training the model ({job['training']['training_steps']} steps)", {"steps": int(job["training"]["training_steps"])}, status="training")
    train_dir = job_dir / "trained_model"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    run_microwakeword(config_path)

    stage("converting", "Converting", "find_model", "Looking for the generated TFLite model", status="converting")
    tflite = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not tflite.exists():
        raise RuntimeError(f"Expected TFLite model not found at {tflite}")
    model_name = f"{job['slug']}.tflite"
    shutil.copyfile(tflite, job_dir / model_name)
    roc = train_dir / "tflite_stream_state_internal_quant" / "tflite_streaming_roc.txt"
    summary, cutoff = summarize_roc(*parse_roc(roc.read_text())) if roc.exists() else (None, 0.97)
    if summary:
        emit("final_metrics", summary=summary)
        log(f"Test set ROC: AUC {summary['auc']}; cutoff {summary['cutoff']:.2f} -> FRR {summary['frr'] * 100:.0f} %, FAPH {summary['faph']:.2f}")
    log(f"Manifest probability_cutoff set to {cutoff:.2f} (tune it in the JSON manifest if the word triggers too easily / too rarely)")
    write_manifest(job, job_dir, clip_ms, model_name, cutoff)
    # free disk: the per-job feature mmaps are large and only needed during training
    shutil.rmtree(features_dir, ignore_errors=True)
    kb = (job_dir / model_name).stat().st_size // 1024
    minutes = round((time.time() - started) / 60, 1)
    emit("done", message_key="model_ready", params={"name": model_name, "kb": kb, "minutes": minutes}, message=f"Model {model_name} ({kb} kB) ready in {minutes} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        emit("stage", key="failed", name="Error", message=str(exc), params={})
        sys.exit(1)
