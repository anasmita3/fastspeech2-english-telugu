"""
Choose ~5 h train + ~1 h validation from a Hugging Face audio dataset and write it
in the folder layout the aligner / FastSpeech 2 preprocessor expects:

    raw_data/<dataset>/<speaker>/<utt>.wav   (22.05 kHz, mono, 16-bit)
    raw_data/<dataset>/<speaker>/<utt>.lab   (normalised transcript, one line)
    raw_data/<dataset>/split.json            {"val": ["speaker|utt", ...]}
    raw_data/<dataset>/selection_stats.json  (numbers for the report)

Selection strategy (written down here because the assignment asks for it):
  1. Keep clips whose length is within [min_seconds, max_seconds] and whose
     transcript is non-empty after normalisation.
  2. Drop speakers with less than `min_speaker_seconds` of usable audio.
  3. Every remaining speaker contributes the SAME FRACTION of its audio
     (fraction = wanted_hours / available_hours), picked at random with a fixed
     seed, so no speaker dominates.
  4. Inside each speaker the picked clips are split into train / validation
     with the ratio train_hours : val_hours. The split is by utterance, so
     train and validation never share a clip.
"""
import io
import json
import os
import random
from collections import defaultdict

import numpy as np

TEXT_CANDIDATES = ["text", "sentence", "transcript", "transcription", "normalized_text",
                   "normalized", "verbatim", "utterance"]
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "spk_id", "client_id", "speaker_name", "gender"]
SPEAKER_META = ["gender", "age-group", "primary_language", "highest_qualification", "job_category", "occupation_domain"]
# Svarah has no speaker-id column; these metadata columns together almost identify a speaker
SPEAKER_META = ["gender", "age-group", "primary_language", "highest_qualification", "job_category",
                "occupation_domain"]


def detect_columns(ds, text_col=None, speaker_col=None):
    from datasets import Audio

    cols = list(ds.features)
    audio_col = next((c for c in cols if isinstance(ds.features[c], Audio)), None)
    if audio_col is None:
        raise ValueError("No audio column found. Columns are: {}".format(cols))
    text_col = text_col or next((c for c in TEXT_CANDIDATES if c in cols), None)
    if text_col is None:
        raise ValueError("Could not find the text column. Columns are: {}. "
                         "Set data.text_column in config.yaml.".format(cols))
    speaker_col = speaker_col or next((c for c in SPEAKER_CANDIDATES if c in cols), None)
    return audio_col, text_col, speaker_col


def speaker_meta_columns(ds, speaker_col):
    if speaker_col not in (None, "gender"):
        return []
    cols = list(ds.features)
    meta = [c for c in cols if c in SPEAKER_META or c.startswith("native_")]
    return meta if len(meta) > 1 else []


def speaker_meta_columns(ds, speaker_col):
    """If the only 'speaker' column is weak (gender) or missing, combine metadata columns into a
    pseudo-speaker key. Returns [] when a real speaker-id column exists."""
    if speaker_col not in (None, "gender"):
        return []
    cols = list(ds.features)
    meta = [c for c in cols if c in SPEAKER_META or c.startswith("native_")]
    return meta if len(meta) > 1 else []


def _decode(audio_dict):
    import soundfile as sf

    if audio_dict.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(audio_dict["bytes"]), dtype="float32", always_2d=False)
    else:
        wav, sr = sf.read(audio_dict["path"], dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def select_and_export(ds, out_dir, normalize_fn, train_hours=5.0, val_hours=1.0, seed=1234,
                      min_seconds=1.0, max_seconds=12.0, min_speaker_seconds=60,
                      sampling_rate=22050, text_col=None, speaker_col=None, log=print):
    import librosa
    from datasets import Audio
    from scipy.io import wavfile

    audio_col, text_col, speaker_col = detect_columns(ds, text_col, speaker_col)
    log("columns -> audio: {!r}, text: {!r}, speaker: {!r}".format(audio_col, text_col, speaker_col))
    meta_cols = speaker_meta_columns(ds, speaker_col)
    spk_keys = {}
    if meta_cols:
        log("no speaker-id column: pseudo-speakers = unique combinations of {}".format(meta_cols))
    elif speaker_col is None:
        log("WARNING: no speaker column found; treating the corpus as ONE speaker.")
    ds = ds.cast_column(audio_col, Audio(decode=False))  # we decode ourselves (no ffmpeg needed)

    # ---- pass 1: cheap metadata only (needs decoding to know the real duration)
    rng = random.Random(seed)
    items = []
    dropped = defaultdict(int)
    for idx in range(len(ds)):
        row = ds[idx]
        text = normalize_fn(str(row[text_col] or ""))
        if not text:
            dropped["empty text"] += 1
            continue
        wav, sr = _decode(row[audio_col])
        dur = len(wav) / sr
        if dur < min_seconds:
            dropped["too short"] += 1
            continue
        if dur > max_seconds:
            dropped["too long"] += 1
            continue
        if meta_cols:
            key = tuple(str(row[c]) for c in meta_cols)
            if key not in spk_keys:
                spk_keys[key] = "spk{:03d}".format(len(spk_keys))
            spk = spk_keys[key]
        else:
            spk = _safe_name(row[speaker_col]) if speaker_col else "spk0"
        items.append({"idx": idx, "speaker": spk, "dur": dur, "text": text})
    log("usable clips: {} ({:.2f} h); dropped: {}".format(
        len(items), sum(i["dur"] for i in items) / 3600, dict(dropped)))

    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)
    kept_spk = {s: v for s, v in by_spk.items() if sum(i["dur"] for i in v) >= min_speaker_seconds}
    log("speakers: {} total, {} kept (>= {} s usable audio)".format(len(by_spk), len(kept_spk), min_speaker_seconds))
    avail = sum(i["dur"] for v in kept_spk.values() for i in v)
    want = (train_hours + val_hours) * 3600
    frac = min(1.0, want / avail)
    val_share = val_hours / (train_hours + val_hours)
    if frac < 1.0 or avail < want:
        log("available {:.2f} h, wanted {:.2f} h -> every speaker contributes {:.1%} of its audio".format(
            avail / 3600, want / 3600, frac))
    if avail < want:
        log("NOTE: less audio available than requested; using everything that is usable.")

    train, val = [], []
    for spk in sorted(kept_spk):
        v = kept_spk[spk][:]
        rng.shuffle(v)
        quota = frac * sum(i["dur"] for i in v)
        picked, tot = [], 0.0
        for it in v:
            if tot >= quota:
                break
            picked.append(it)
            tot += it["dur"]
        n_val_sec = val_share * tot
        acc = 0.0
        for it in picked:
            if acc < n_val_sec:
                val.append(it)
                acc += it["dur"]
            else:
                train.append(it)

    # ---- pass 2: write files
    os.makedirs(out_dir, exist_ok=True)
    val_keys = []
    stats = {"speakers": defaultdict(lambda: {"train_s": 0.0, "val_s": 0.0, "train_n": 0, "val_n": 0})}
    for part, lst in (("train", train), ("val", val)):
        for it in lst:
            row = ds[it["idx"]]
            wav, sr = _decode(row[audio_col])
            if sr != sampling_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
            peak = max(1e-6, float(np.max(np.abs(wav))))
            wav16 = (wav / peak * 0.95 * 32767).astype(np.int16)
            utt = "{}_{:06d}".format(it["speaker"], it["idx"])
            spk_dir = os.path.join(out_dir, it["speaker"])
            os.makedirs(spk_dir, exist_ok=True)
            wavfile.write(os.path.join(spk_dir, utt + ".wav"), sampling_rate, wav16)
            with open(os.path.join(spk_dir, utt + ".lab"), "w", encoding="utf-8") as f:
                f.write(it["text"] + "\n")
            if part == "val":
                val_keys.append("{}|{}".format(it["speaker"], utt))
            s = stats["speakers"][it["speaker"]]
            s[part + "_s"] += it["dur"]
            s[part + "_n"] += 1

    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_keys}, f)
    summary = {
        "train_hours": sum(i["dur"] for i in train) / 3600,
        "val_hours": sum(i["dur"] for i in val) / 3600,
        "train_utterances": len(train),
        "val_utterances": len(val),
        "n_speakers": len(stats["speakers"]),
        "dropped": dict(dropped),
        "pseudo_speaker_columns": meta_cols,
        "pseudo_speaker_keys": {v: list(k) for k, v in spk_keys.items()},
        "settings": {"train_hours": train_hours, "val_hours": val_hours, "seed": seed,
                     "min_seconds": min_seconds, "max_seconds": max_seconds,
                     "min_speaker_seconds": min_speaker_seconds},
        "per_speaker": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                        for k, v in stats["speakers"].items()},
    }
    with open(os.path.join(out_dir, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # the train and val sets must not share a clip
    assert not (set("{}|{}_{:06d}".format(i["speaker"], i["speaker"], i["idx"]) for i in train) & set(val_keys))
    log("exported: train {:.2f} h ({} clips), val {:.2f} h ({} clips), {} speakers".format(
        summary["train_hours"], len(train), summary["val_hours"], len(val), summary["n_speakers"]))
    return summary
