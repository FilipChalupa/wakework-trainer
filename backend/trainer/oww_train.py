"""openWakeWord-compatible training (target "wyoming"): speech-embedding features -> small DNN classifier -> .tflite
that wyoming-openwakeword and the Home Assistant openWakeWord add-on load as a custom model.

Mirrors openwakeword.train.Model (Flatten -> Dense -> LayerNorm -> ReLU -> block -> Dense -> sigmoid) in Keras, so
no torch / onnx-tf is needed. Positive examples are the last 16 embedding frames of augmented wake word clips, negatives
come from the shared datasets (speech commands composites, ESC-50, FMA), synthetic noise, the user's negative
recordings and wake word fragments (hard negatives). False accepts per hour are estimated on long ambient streams and,
when downloaded, on openWakeWord's own 10-hour validation feature set.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from trainer.audio_data import SR, WavClips, build_ambient_clip, load_wav, peak_normalize, stable_split
from trainer.oww_features import CLASSIFIER_FRAMES, EMBEDDING_DIM, OwwFeatures, classifier_windows, ensure_oww_models

FEATURE_VERSION = "v1"


def _run_helpers():
    from trainer import run

    return run


def last_window(embeddings: np.ndarray) -> np.ndarray | None:
    if embeddings.shape[0] < CLASSIFIER_FRAMES:
        return None
    return embeddings[-CLASSIFIER_FRAMES:].astype(np.float32)


def all_windows(embeddings: np.ndarray, step: int) -> list[np.ndarray]:
    return [embeddings[a:b].astype(np.float32) for a, b in classifier_windows(embeddings.shape[0], step)]


def build_model(layer_dim: int = 128, n_blocks: int = 1):
    import tensorflow as tf

    inputs = tf.keras.Input(batch_shape=(None, CLASSIFIER_FRAMES, EMBEDDING_DIM), name="input")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(layer_dim)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(n_blocks):
        x = tf.keras.layers.Dense(layer_dim)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs)


def export_tflite(model, path: Path) -> None:
    """Float32 TFLite with a fixed (1, 16, 96) input – the shape wyoming-openwakeword expects."""
    import tensorflow as tf

    fixed = build_model(layer_dim=model.layers[2].units, n_blocks=(len(model.layers) - 5) // 3)
    fixed.set_weights(model.get_weights())
    inputs = tf.keras.Input(batch_shape=(1, CLASSIFIER_FRAMES, EMBEDDING_DIM), name="input")
    single = tf.keras.Model(inputs, fixed(inputs))
    converter = tf.lite.TFLiteConverter.from_keras_model(single)
    converter.optimizations = []
    path.write_bytes(converter.convert())


def false_accepts(probs: np.ndarray, cutoff: float, refractory: int = 12) -> int:
    """Counts activations on a probability stream with a refractory period (like the streaming detector)."""
    count = 0
    cooldown = 0
    for p in probs:
        if cooldown > 0:
            cooldown -= 1
            continue
        if p >= cutoff:
            count += 1
            cooldown = refractory
    return count


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# --------------------------------------------------------------------------------------
def _embed_generator(feats: OwwFeatures, generator, expected: int, label: str, take: str = "last", step: int = 8, log=None, emit=None) -> np.ndarray:
    out = []
    last = time.time()
    n = 0
    for audio in generator:
        n += 1
        emb = feats.embed_clip(audio)
        if take == "last":
            w = last_window(emb)
            if w is not None:
                out.append(w)
        else:
            out.extend(all_windows(emb, step))
        if emit and (n % 25 == 0 or time.time() - last > 1.0):
            last = time.time()
            emit("progress", current=n, total=expected)
    arr = np.asarray(out, dtype=np.float32) if out else np.zeros((0, CLASSIFIER_FRAMES, EMBEDDING_DIM), dtype=np.float32)
    if log:
        log(f"{label}: {arr.shape[0]} examples")
    return arr


def _files_digest(paths: list[Path]) -> str:
    """Short hash of the file list (names + sizes) so a re-downloaded dataset invalidates the feature cache."""
    import hashlib

    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            h.update(f"{p.name}:{p.stat().st_size}".encode())
        except OSError:
            continue
    return h.hexdigest()[:10]


def _cached(cache_dir: Path, name: str, builder, source_files: list[Path] | None = None) -> np.ndarray:
    if source_files:
        name = f"{name}_{_files_digest(source_files)}"
    path = cache_dir / f"{name}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    arr = builder()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def run_wyoming(job: dict, job_dir: Path, features_dir: Path, started: float, noise_files: list[Path], background_dirs: list[str], real_backgrounds: list[Path]) -> None:
    run = _run_helpers()
    stage, log, emit = run.stage, run.log, run.emit
    training = job["training"]
    rng = random.Random(11)

    stage("preparing", "Preparing data", "oww_models", "Downloading openWakeWord feature models", status="preparing")
    models_dir = ensure_oww_models(Path(job["feature_cache_dir"]))
    feats = OwwFeatures(models_dir, threads=4)
    cache_dir = Path(job["feature_cache_dir"]) / f"oww_features_{FEATURE_VERSION}"
    datasets_dir = Path(job["datasets_dir"])

    # ----- positives -----
    positive_paths = sorted(Path(job["positive_dir"]).glob("*.wav"))
    if len(positive_paths) < 3:
        raise RuntimeError("At least 3 positive samples are required")
    splits = stable_split(positive_paths)
    clips = WavClips(splits, trim=True)
    durations = clips.durations("train") + clips.durations("validation") + clips.durations("test")
    p90 = float(np.percentile(durations, 90))
    clip_s = 2.0 if p90 + 0.45 <= 2.0 else float(np.ceil((p90 + 0.45) * 2) / 2)  # 2.0 s -> exactly 16 embedding frames
    log(f"Positive samples: {len(positive_paths)} (train {len(splits['train'])}, val {len(splits['validation'])}, test {len(splits['test'])}); clip {clip_s:.1f} s")
    reps = max(1, int(training["augmentations_per_sample"]))
    pos = {}
    for set_name, split, repeat in (("training", "train", reps), ("validation", "validation", max(2, reps // 4)), ("testing", "test", max(2, reps // 4))):
        expected = len(splits[split]) * repeat
        stage("preparing", "Preparing data", "augment_positive", f"Augmenting positive samples ({set_name})", {"set": set_name}, current=0, total=expected)
        augmenter = run.make_augmenter(clip_s, background_dirs, positive=True)
        gen = augmenter.augment_generator(clips.audio_generator(split=split, repeat=repeat))
        pos[set_name] = _embed_generator(feats, gen, expected, f"oww positive/{set_name}", take="last", log=log, emit=emit)

    # ----- hard negatives from word fragments -----
    hard = np.zeros((0, CLASSIFIER_FRAMES, EMBEDDING_DIM), dtype=np.float32)
    if training.get("hard_negatives", True):
        source = run.HardNegativeClips(clips)
        variants = source._variants("train")
        expected = len(variants) * max(2, reps // 4)
        stage("preparing", "Preparing data", "hard_negatives", "Hard negatives from partial wake words (training)", {"set": "training"}, current=0, total=expected)
        augmenter = run.make_augmenter(clip_s, background_dirs, positive=True)
        gen = augmenter.augment_generator(source.audio_generator("train", repeat=max(2, reps // 4)))
        hard = _embed_generator(feats, gen, expected, "oww hard negatives", take="last", log=log, emit=emit)

    # ----- shared negative features (cached) -----
    speech = run.list_speech_commands(job)
    negatives = []

    def speech_builder():
        picked = speech
        composites = 3000
        stage("preparing", "Preparing data", "speech_features", f"Speech-command composites ({composites} clips)", {"set": "training", "count": composites}, current=0, total=composites)

        def gen():
            r = random.Random(3)
            for _ in range(composites):
                yield build_ambient_clip(picked, 2.0, r, noise_files)

        return _embed_generator(feats, gen(), composites, "oww negatives/speech", take="last", log=log, emit=emit)

    if speech:
        negatives.append(("speech", _cached(cache_dir, "speech_commands", speech_builder, speech)))

    for name, folder, step in (("esc50", datasets_dir / "esc50", 12), ("fma", datasets_dir / "fma_16k", 16)):
        wavs = sorted(folder.rglob("*.wav")) if folder.is_dir() else []
        if not wavs:
            continue

        def builder(wavs=wavs, name=name, step=step):
            stage("preparing", "Preparing data", "noise_features", f"Background dataset features ({name})", {"set": name}, current=0, total=len(wavs))

            def gen():
                for w in wavs:
                    yield load_wav(w)

            return _embed_generator(feats, gen(), len(wavs), f"oww negatives/{name}", take="all", step=step, log=log, emit=emit)

        negatives.append((name, _cached(cache_dir, name, builder, wavs)))

    # per-project negatives: synthetic noise crops + the user's negative recordings
    user_negatives = sorted(Path(job["negative_dir"]).glob("*.wav"))
    stage("preparing", "Preparing data", "noise_features", "Noise and custom negative recordings (training)", {"set": "training"}, current=0, total=len(noise_files) * 12 + len(user_negatives) * 10)

    def project_neg_gen():
        noise_clips = WavClips({"train": noise_files, "validation": noise_files, "test": noise_files}, trim=False, normalize=False)
        aug = run.make_augmenter(clip_s, [], positive=False)
        yield from aug.augment_generator(noise_clips.audio_generator("train", repeat=12))
        if user_negatives:
            user_clips = WavClips({"train": user_negatives, "validation": user_negatives, "test": user_negatives}, trim=False)
            aug2 = run.make_augmenter(clip_s, [str(noise_files[0].parent)], positive=False)
            yield from aug2.augment_generator(user_clips.audio_generator("train", repeat=10))

    negatives.append(("project", _embed_generator(feats, project_neg_gen(), len(noise_files) * 12 + len(user_negatives) * 10, "oww negatives/project", take="last", log=log, emit=emit)))

    # ----- ambient streams for false accepts per hour -----
    stage("preparing", "Preparing data", "ambient_features", "Long ambient recording for false-accept measurement (validation_ambient)", {"set": "validation_ambient"}, current=0, total=4)
    pool = list(stable_split(speech)["validation"]) + user_negatives * 5 if speech else list(user_negatives) or noise_files
    beds = noise_files + (rng.sample(real_backgrounds, min(40, len(real_backgrounds))) if real_backgrounds else [])
    ambient = {}
    for i, (set_name, seed) in enumerate((("validation", 7), ("testing", 11))):
        r = random.Random(seed)
        stream_pool = pool + (r.sample(real_backgrounds, min(150, len(real_backgrounds))) if real_backgrounds else [])
        emb = np.concatenate([feats.embed_clip(build_ambient_clip(stream_pool, 120.0, r, beds)) for _ in range(2)])
        ambient[set_name] = emb
        emit("progress", current=2 * (i + 1), total=4)
    oww_val = datasets_dir / "oww_validation" / "validation_set_features.npy"
    if oww_val.exists():
        big = np.load(oww_val, mmap_mode="r")
        half = big.shape[0] // 2
        ambient["validation"] = np.concatenate([ambient["validation"], np.asarray(big[: min(half, 60000)])])
        ambient["testing"] = np.concatenate([ambient["testing"], np.asarray(big[half : half + min(half, 60000)])])
        log(f"Using openWakeWord validation features: {big.shape[0]} frames")

    # ----- assemble -----
    def split_holdout(arr: np.ndarray, frac: float = 0.1):
        n = arr.shape[0]
        if n < 20:
            return arr, arr[:0]
        idx = np.arange(n)
        random.Random(5).shuffle(idx)
        k = max(1, int(n * frac))
        return arr[idx[k:]], arr[idx[:k]]

    neg_train_parts, neg_val_parts = [], []
    for name, arr in negatives:
        tr, va = split_holdout(np.asarray(arr))
        neg_train_parts.append(tr)
        neg_val_parts.append(va)
        log(f"Negatives {name}: {tr.shape[0]} train / {va.shape[0]} val")
    if hard.shape[0]:
        tr, va = split_holdout(hard)
        neg_train_parts.append(tr)
        neg_val_parts.append(va)
    X_neg = np.concatenate(neg_train_parts) if neg_train_parts else np.zeros((0, CLASSIFIER_FRAMES, EMBEDDING_DIM), np.float32)
    X_neg_val = np.concatenate(neg_val_parts) if neg_val_parts else X_neg[:0]
    X_pos, X_pos_val, X_pos_test = pos["training"], pos["validation"], pos["testing"]
    if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
        raise RuntimeError("Not enough examples to train (positives or negatives empty)")
    log(f"Training set: {X_pos.shape[0]} positive / {X_neg.shape[0]} negative windows; validation {X_pos_val.shape[0]} / {X_neg_val.shape[0]}; ambient frames {ambient['validation'].shape[0]}")
    log(f"Data preparation finished in {time.time() - started:.0f}s")

    # ----- training -----
    import tensorflow as tf

    steps = int(training["training_steps"])
    batch = int(training["batch_size"])
    eval_interval = max(25, min(int(training["eval_step_interval"]), steps))
    emit("training_config", total_steps=steps, eval_step_interval=eval_interval, clip_duration_ms=int(clip_s * 1000))
    stage("training", "Training", "train_steps", f"Training the openWakeWord classifier ({steps} steps)", {"steps": steps}, status="training")
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(float(training["learning_rate"]) * 10)  # openWakeWord uses 1e-4 on batches of 1024 for 50k+ steps; we train fewer steps
    bce = tf.keras.losses.BinaryCrossentropy()
    neg_weight = float(min(4.0, max(1.0, float(training["negative_class_weight"]) / 5.0)))
    pos_frac = 0.4
    n_pos_batch = max(1, int(batch * pos_frac))
    n_neg_batch = batch - n_pos_batch
    train_dir = job_dir / "trained_model"
    train_dir.mkdir(parents=True, exist_ok=True)
    best_score = None
    best_weights = model.get_weights()

    @tf.function
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            loss = bce(y, p, sample_weight=w)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, p

    def predict(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        return model.predict(x, batch_size=2048, verbose=0).reshape(-1)

    def validate(step: int) -> dict:
        p_pos = predict(X_pos_val)
        p_neg = predict(X_neg_val)
        stream = predict(np.asarray(all_windows(ambient["validation"], 1)))
        hours = ambient["validation"].shape[0] * 0.08 / 3600.0
        tp = int((p_pos >= 0.5).sum())
        fp = int((p_neg >= 0.5).sum())
        recall = tp / max(1, p_pos.size)
        precision = tp / max(1, tp + fp)
        accuracy = (tp + int((p_neg < 0.5).sum())) / max(1, p_pos.size + p_neg.size)
        cutoffs = np.linspace(0.0, 1.0, 101)
        faph = np.array([false_accepts(stream, c) / max(hours, 1e-6) for c in cutoffs])
        recalls = np.array([(p_pos >= c).mean() if p_pos.size else 0.0 for c in cutoffs])
        viable = faph <= 2.0
        avg_viable_recall = float(recalls[viable].mean()) if viable.any() else 0.0
        no_fa = np.where(faph == 0)[0]
        cutoff_no_fa = float(cutoffs[no_fa[0]]) if no_fa.size else 1.0
        recall_no_fa = float(recalls[no_fa[0]]) if no_fa.size else 0.0
        eps = 1e-7
        loss = float(-(np.log(p_pos + eps).sum() + np.log(1 - p_neg + eps).sum()) / max(1, p_pos.size + p_neg.size))
        entry = {
            "step": step,
            "recall_at_no_faph": recall_no_fa,
            "cutoff_for_no_faph": cutoff_no_fa,
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "ambient_false_positives": int(false_accepts(stream, 0.5)),
            "false_positives_per_hour": float(false_accepts(stream, 0.5) / max(hours, 1e-6)),
            "loss": loss,
            "auc": auc_score(p_pos, p_neg),
            "average_viable_recall": avg_viable_recall,
        }
        emit("validation", entry=entry)
        log(f"Step {step}: val recall {recall:.3f}, precision {precision:.3f}, FA/h@0.5 {entry['false_positives_per_hour']:.2f}, recall@noFA {recall_no_fa:.3f} (cutoff {cutoff_no_fa:.2f}), avg viable recall {avg_viable_recall:.3f}")
        return entry

    rng_np = np.random.default_rng(1)
    for step in range(1, steps + 1):
        xi = rng_np.integers(0, X_pos.shape[0], n_pos_batch)
        ni = rng_np.integers(0, X_neg.shape[0], n_neg_batch)
        x = np.concatenate([X_pos[xi], X_neg[ni]]).astype(np.float32)
        y = np.concatenate([np.ones(n_pos_batch), np.zeros(n_neg_batch)]).astype(np.float32)[:, None]
        w = np.concatenate([np.ones(n_pos_batch), np.full(n_neg_batch, neg_weight)]).astype(np.float32)
        if step == int(steps * 0.7):
            optimizer.learning_rate.assign(float(training["learning_rate"]))
        loss, p = train_step(tf.constant(x), tf.constant(y), tf.constant(w))
        if step % 20 == 0 or step == steps:
            p = p.numpy().reshape(-1)
            pred = p >= 0.5
            tp = int((pred[:n_pos_batch]).sum())
            fp = int((pred[n_pos_batch:]).sum())
            emit("training_step", step=step, total=steps, loss=float(loss), accuracy=float((pred == (y.reshape(-1) >= 0.5)).mean()), recall=tp / n_pos_batch, precision=tp / max(1, tp + fp))
        if step % eval_interval == 0 or step == steps:
            entry = validate(step)
            score = (entry["false_positives_per_hour"] <= 2.0, -entry["false_positives_per_hour"] if entry["false_positives_per_hour"] > 2.0 else 0.0, entry["average_viable_recall"], entry["recall"])
            if best_score is None or score > best_score:
                best_score = score
                best_weights = model.get_weights()
                log(f"New best weights at step {step}")
    model.set_weights(best_weights)
    model.save_weights(str(train_dir / "best_weights.weights.h5"))

    # ----- export + test metrics -----
    stage("converting", "Converting", "find_model", "Exporting the openWakeWord classifier to TFLite", status="converting")
    model_name = f"{job['slug']}.tflite"
    export_tflite(model, job_dir / model_name)
    p_test = predict(X_pos_test)
    stream_test = predict(np.asarray(all_windows(ambient["testing"], 1)))
    hours_test = ambient["testing"].shape[0] * 0.08 / 3600.0
    points = []
    for c in np.linspace(0.05, 0.95, 19):
        points.append({"cutoff": round(float(c), 2), "frr": float(1 - (p_test >= c).mean()) if p_test.size else 1.0, "faph": float(false_accepts(stream_test, c) / max(hours_test, 1e-6))})
    summary, _ = run.summarize_roc(auc_score(p_test, predict(X_neg_val)), points)
    stage("converting", "Converting", "auto_threshold", "Evaluating the model on your recordings to pick the threshold", status="converting")
    auto = run.auto_threshold(job, job_dir / model_name, windows=(1,))
    cutoff = float(auto["cutoff"]) if auto else 0.5
    if auto:
        log(f"Auto threshold {cutoff:.2f}: {auto['positives_passed']}/{auto['positives_total']} positive recordings pass, highest negative {auto['negatives_max']:.2f}")
    summary = dict(summary or {"auc": None, "cutoff": cutoff, "frr": 0.0, "faph": 0.0, "points": points})
    summary["manifest_cutoff"] = cutoff
    summary["auto_threshold"] = auto
    emit("final_metrics", summary=summary)
    (job_dir / f"{job['slug']}.oww.json").write_text(json.dumps({
        "type": "openwakeword",
        "wake_word": job["wake_word"],
        "model": model_name,
        "threshold": cutoff,
        "trigger_level": 1,
        "input": [1, CLASSIFIER_FRAMES, EMBEDDING_DIM],
        "feature_models": "openWakeWord v0.5.1 melspectrogram + embedding_model",
    }, indent=2))
    shutil.rmtree(features_dir, ignore_errors=True)
    kb = (job_dir / model_name).stat().st_size // 1024
    minutes = round((time.time() - started) / 60, 1)
    emit("done", message_key="model_ready", params={"name": model_name, "kb": kb, "minutes": minutes}, message=f"Model {model_name} ({kb} kB) ready in {minutes} min")
