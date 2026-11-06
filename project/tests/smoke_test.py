"""
End-to-end smoke test on FAKE speech (runs on CPU in a few minutes).

It checks that every stage of the pipeline runs and produces the files the
assignment asks for. The Montreal Forced Aligner is replaced by exact
alignments (we know how long every fake phone is), so it does NOT test the
aligner and does NOT test audio quality.

    python tests/smoke_test.py --language telugu
    python tests/smoke_test.py --language english
"""
import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.configio import setup_paths  # noqa: E402

setup_paths()

SR = 22050
TELUGU = ["ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది.", "పులి అడవిలో నడిచింది.",
          "ఆమె పట్టు చీరను ధరించి వచ్చింది.", "తాడుపై అతని పట్టు బలంగా ఉంది.", "న్యూరోఫార్మకోలాజికల్ పరిశోధన జరిగింది.",
          "నూతన ఔషధాల ప్రభావం చాలా ఉంది.", "సమయానికి సిద్ధంగా ఉంచండి.", "వేడుకకు అందరూ వచ్చారు.",
          "అతని పట్టు చాలా బలంగా ఉంది, ఆమె వచ్చింది."]
ENGLISH = ["The meeting will start at 10:45 AM on 12 November 2026.", "The orchestra played during the ceremony.",
           "The researchers will conduct an experiment.", "The rhythm of the music changed.",
           "She walked into the old library, and then she sat down.", "We should arrive on time for the meeting.",
           "The movement gained momentum during the proceedings.", "He was praised by the audience.",
           "A quick brown fox jumps over the lazy dog."]


def fake_wave(phones, rng, f0_base):
    """Voiced tone per phone (frequency depends on the phone), 70-130 ms each."""
    segs, durs = [], []
    for p in phones:
        d = rng.integers(7, 14) * 256  # samples (multiple of the hop length)
        t = np.arange(d) / SR
        if p in ("sp", "spn", "sil"):
            seg = np.zeros(d)
        else:
            f = f0_base * (1 + 0.04 * (hash(p) % 12))
            seg = 0.3 * np.sin(2 * np.pi * f * t) + 0.1 * np.sin(2 * np.pi * 2 * f * t) + 0.01 * rng.standard_normal(d)
        segs.append(seg)
        durs.append(d)
    return np.concatenate(segs), durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["telugu", "english"])
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    L = a.language
    import soundfile as sf
    import tgt
    from datasets import Audio, Dataset, Features, Value

    if L == "english":
        from common import english_frontend as fe
        norm, phones_of = fe.clean_for_training, lambda t: fe.text_to_phones(t)
        sents = ENGLISH
    else:
        from common import telugu_frontend as fe
        norm, phones_of = fe.normalize_text, lambda t: fe.text_to_phones(t)
        sents = TELUGU

    tmp = os.path.abspath("/tmp/fs2_smoke_" + L)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    proj = os.path.join(tmp, "project")
    shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns("work", "__pycache__", "*.pth", "*.pth.tar", "tests", "*.zip"))
    # keep the hifigan weights (zip) from the real project
    shutil.copytree(os.path.join(ROOT, "fastspeech2", "hifigan"), os.path.join(proj, "fastspeech2", "hifigan"), dirs_exist_ok=True)

    lang_dir = os.path.join(proj, L.capitalize())
    cfg_path = os.path.join(lang_dir, "config.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["data"].update(train_hours=0.2, val_hours=0.04, min_speaker_seconds=5, min_seconds=0.5)
    cfg["train"]["optimizer"].update(batch_size=4, warm_up_step=20, anneal_steps=[40, 50])
    cfg["train"]["step"].update(total_step=a.steps, log_step=10, val_step=20, save_step=30)
    cfg["model"]["transformer"].update(encoder_layer=2, decoder_layer=2, conv_filter_size=256,
                                       encoder_hidden=128, decoder_hidden=128)
    cfg["model"]["variance_predictor"]["filter_size"] = 128
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    work = os.path.join(tmp, "work")
    os.environ["FS2_WORK"] = work

    # ---- 1. fake Hugging-Face-style dataset (3 speakers, ~140 clips)
    rng = np.random.default_rng(0)
    rows = {"audio": [], "text": [], "speaker_id": []}
    for spk, f0 in (("spkA", 120), ("spkB", 210), ("spkC", 160)):
        for k in range(48):
            t = sents[(k + len(spk)) % len(sents)]
            wav, _ = fake_wave(phones_of(norm(t)), rng, f0)
            buf = io.BytesIO()
            sf.write(buf, wav.astype(np.float32), SR, format="WAV")
            rows["audio"].append({"bytes": buf.getvalue(), "path": None})
            rows["text"].append(t)
            rows["speaker_id"].append(spk)
    class FakeDS:
        """Minimal stand-in for a Hugging Face Dataset (the real one needs torchcodec just to *build* audio)."""
        features = {"audio": Audio(decode=False), "text": Value("string"), "speaker_id": Value("string")}

        def __len__(self):
            return len(rows["text"])

        def cast_column(self, *a, **k):
            return self

        def __getitem__(self, i):
            return {c: rows[c][i] for c in rows}

    ds = FakeDS()
    # stage 'select' (offline: uses the dataset object instead of downloading)
    from common.configio import load_config
    from common.data_select import select_and_export
    from common import pipeline

    cfg2, configs = load_config(cfg_path)
    d = cfg2["data"]
    summ = select_and_export(ds, cfg2["preprocess"]["path"]["raw_path"], pipeline._normalizer(L), train_hours=d["train_hours"],
                             val_hours=d["val_hours"], seed=d["seed"], min_seconds=d["min_seconds"],
                             max_seconds=d["max_seconds"], min_speaker_seconds=d["min_speaker_seconds"])
    assert summ["train_utterances"] > 40 and summ["val_utterances"] > 5, summ

    # ---- 2. lexicon stage (real code)
    pipeline.stage_lexicon(L, cfg2)

    # ---- 3. stand-in for the aligner: exact TextGrids (set SMOKE_USE_MFA=1 to run the REAL aligner instead)
    raw = cfg2["preprocess"]["path"]["raw_path"]
    if os.environ.get("SMOKE_USE_MFA"):
        pipeline.stage_align(L, cfg2, 2, fast=bool(os.environ.get("SMOKE_FAST")))
        pipeline.stage_features(L, cfg2)
        print("REAL ALIGNER PATH OK")
        return
    tg_root = os.path.join(cfg2["preprocess"]["path"]["preprocessed_path"], "TextGrid")
    n = 0
    for lab in glob.glob(os.path.join(raw, "*", "*.lab")):
        spk, base = os.path.basename(os.path.dirname(lab)), os.path.basename(lab)[:-4]
        text = open(lab, encoding="utf-8").read().strip()
        ph = phones_of(text) if L == "telugu" else phones_of(text)
        wav, sr = sf.read(os.path.join(raw, spk, base + ".wav"))
        # re-create the same random durations: equal split is good enough for a smoke test
        per = len(wav) / len(ph) / SR
        tier = tgt.IntervalTier(start_time=0, end_time=len(wav) / SR, name="phones")
        for i, p in enumerate(ph):
            tier.add_interval(tgt.Interval(i * per, (i + 1) * per if i < len(ph) - 1 else len(wav) / SR, p))
        tg = tgt.TextGrid()
        tg.add_tier(tier)
        os.makedirs(os.path.join(tg_root, spk), exist_ok=True)
        tgt.io.write_to_file(tg, os.path.join(tg_root, spk, base + ".TextGrid"), format="long")
        n += 1
    print("fake TextGrids:", n)

    # ---- 4. features stage (real code)
    pipeline.stage_features(L, cfg2)

    # ---- 5. training via the real CLI
    env = dict(os.environ)
    py = sys.executable
    r = subprocess.run([py, os.path.join(lang_dir, "train.py"), "--plots_dir", os.path.join(proj, "Results", L.capitalize())],
                       env=env)
    assert r.returncode == 0, "train.py failed"
    # resume check: asking for more steps must continue from the saved step
    r = subprocess.run([py, os.path.join(lang_dir, "train.py"), "--total_step", str(a.steps + 10),
                        "--plots_dir", os.path.join(proj, "Results", L.capitalize())], env=env)
    assert r.returncode == 0, "resume failed"

    # ---- 6. inference via the real CLI
    r = subprocess.run([py, os.path.join(lang_dir, "inference.py"), "--checkpoint", "last", "--all_speakers"], env=env)
    assert r.returncode == 0, "inference.py failed"
    res = os.path.join(proj, "Results", L.capitalize())
    expect = ["training_loss.png", "validation_loss.png", "test_generated_audios", "variance_adaptor_plots/duration",
              "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy", "variance_adaptor_values.json"]
    for e in expect:
        assert os.path.exists(os.path.join(res, e)), "missing " + e
    wavs = sorted(glob.glob(os.path.join(res, "test_generated_audios", "*.wav")))
    assert len(wavs) == 5, wavs
    for sub in ("duration", "pitch", "energy"):
        assert len(glob.glob(os.path.join(res, "variance_adaptor_plots", sub, "*.png"))) == 5
    print("\nSMOKE TEST PASSED for", L, "->", res)


if __name__ == "__main__":
    main()
