"""
Glue code shared by English/ and Telugu/ (their preprocess.py, train.py and
inference.py are thin wrappers around the functions in this file).

preprocessing stages (run in this order, or --stage all):
  select    download dataset from Hugging Face, pick ~5 h train / ~1 h val, write wav + .lab
  lexicon   build the pronunciation lexicon the aligner needs
  align     Montreal Forced Aligner -> phone-level TextGrids (who says which phone when)
  features  mel / pitch / energy / duration  (the original repo's Preprocessor)
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from common.configio import load_config, setup_paths

setup_paths()


# ------------------------------------------------------------------ helpers
def _frontend(language):
    if language == "english":
        from common import english_frontend as fe
    else:
        from common import telugu_frontend as fe
    return fe


def _normalizer(language):
    fe = _frontend(language)
    return fe.clean_for_training if language == "english" else fe.normalize_text


def find_mfa():
    exe = shutil.which("mfa")
    if exe:
        return [exe]
    mm_root = os.path.expanduser(os.environ.get("MAMBA_ROOT_PREFIX", "~/micromamba"))
    mm = os.path.join(mm_root, "bin", "micromamba")
    if os.path.exists(mm):
        return [mm, "run", "-r", mm_root, "-n", "mfa", "mfa"]
    raise RuntimeError("Montreal Forced Aligner not found. Run the 'install aligner' cell of the notebook first.")


def read_labs(raw_dir):
    texts = []
    for p in glob.glob(os.path.join(raw_dir, "*", "*.lab")):
        with open(p, encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts


# ------------------------------------------------------------------- stages
def stage_select(language, cfg, hf_token):
    from datasets import concatenate_datasets, load_dataset

    from common.data_select import select_and_export

    d = cfg["data"]
    raw = cfg["preprocess"]["path"]["raw_path"]
    print("loading", d["hf_dataset"], "...")
    ds_dict = load_dataset(d["hf_dataset"], token=hf_token or None)
    print("splits:", {k: len(v) for k, v in ds_dict.items()})
    ds = concatenate_datasets([ds_dict[k] for k in ds_dict])
    print("columns:", list(ds.features))
    return select_and_export(
        ds, raw, _normalizer(language), train_hours=d["train_hours"], val_hours=d["val_hours"],
        seed=d["seed"], min_seconds=d["min_seconds"], max_seconds=d["max_seconds"],
        min_speaker_seconds=d["min_speaker_seconds"],
        sampling_rate=cfg["preprocess"]["preprocessing"]["audio"]["sampling_rate"],
        text_col=d.get("text_column"), speaker_col=d.get("speaker_column"))


def stage_lexicon(language, cfg):
    fe = _frontend(language)
    pp = cfg["preprocess"]["path"]
    texts = read_labs(pp["raw_path"])
    assert texts, "no .lab files found in {} - run --stage select first".format(pp["raw_path"])
    if language == "english":
        base = fe.read_lexicon(pp["base_lexicon"])
        lex = fe.build_lexicon_for_corpus(texts, base)
    else:
        lex = fe.build_lexicon(texts, normalize=False)  # .lab files are already normalised
    fe.write_lexicon(lex, pp["lexicon_path"])
    # every phone must exist in the model's symbol table
    from text.symbols import symbols

    known = set(s[1:] for s in symbols if s.startswith("@"))
    bad = sorted({p for ph in lex.values() for p in ph if p not in known})
    print("lexicon: {} words -> {}".format(len(lex), pp["lexicon_path"]))
    if bad:
        raise ValueError("phones missing from text/symbols.py: {}".format(bad))


FAST_ALIGN_CONFIG = """# faster aligner recipe: monophone + triphone only (the slow lda/sat stages are not needed for TTS durations)
training:
  - monophone:
      num_iterations: 40
      subset: 0
  - triphone:
      num_iterations: 35
      num_leaves: 1000
      max_gaussians: 10000
      subset: 0
"""


def stage_align(language, cfg, num_jobs, fast=False):
    pp = cfg["preprocess"]["path"]
    out_tg = os.path.join(pp["preprocessed_path"], "TextGrid")
    model_zip = os.path.join(cfg["_work"], "mfa_acoustic_model.zip")
    env = dict(os.environ, MFA_ROOT_DIR=os.path.join(cfg["_work"], "mfa_root"))
    extra = []
    if fast:
        cfg_file = os.path.join(cfg["_work"], "mfa_fast_config.yaml")
        with open(cfg_file, "w") as f:
            f.write(FAST_ALIGN_CONFIG)
        extra = ["--config_path", cfg_file]
    cmd = find_mfa() + ["train", "--clean", "-j", str(num_jobs)] + extra + [
        "--output_directory", out_tg, pp["raw_path"], pp["lexicon_path"], model_zip]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    n = len(glob.glob(os.path.join(out_tg, "*", "*.TextGrid")))
    n_wav = len(glob.glob(os.path.join(pp["raw_path"], "*", "*.wav")))
    print("TextGrids written: {} of {} utterances".format(n, n_wav))
    if r.returncode != 0 or n == 0:
        raise RuntimeError("aligner failed (exit code {}). Copy the messages above to get help.".format(r.returncode))


def stage_features(language, cfg):
    from preprocessor.preprocessor import Preprocessor

    pre = cfg["preprocess"]
    Preprocessor(pre).build_from_path()
    # numbers for the report
    import numpy as np

    hop = pre["preprocessing"]["stft"]["hop_length"]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    out = pre["path"]["preprocessed_path"]
    summary = {}
    for part in ("train", "val"):
        secs, n, spks = 0.0, 0, set()
        with open(os.path.join(out, part + ".txt"), encoding="utf-8") as f:
            for line in f:
                b, s = line.split("|")[:2]
                mel = np.load(os.path.join(out, "mel", "{}-mel-{}.npy".format(s, b)), mmap_mode="r")
                secs += mel.shape[0] * hop / sr
                n += 1
                spks.add(s)
        summary[part] = {"utterances": n, "hours": round(secs / 3600, 3), "speakers": len(spks)}
    with open(os.path.join(out, "preprocess_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("after alignment:", json.dumps(summary))


# --------------------------------------------------------------- entry points
def main_preprocess(language, config_path):
    ap = argparse.ArgumentParser(description="{} preprocessing".format(language))
    ap.add_argument("--stage", default="all", choices=["all", "select", "lexicon", "align", "features"])
    ap.add_argument("--work_dir", default=None, help="big temporary folder (default: <language>/work)")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN)")
    ap.add_argument("--num_jobs", type=int, default=4)
    ap.add_argument("--fast_align", action="store_true", help="aligner: monophone+triphone only (much faster)")
    a = ap.parse_args()
    cfg, _ = load_config(config_path, a.work_dir)
    stages = ["select", "lexicon", "align", "features"] if a.stage == "all" else [a.stage]
    for s in stages:
        print("\n===== stage: {} =====".format(s), flush=True)
        if s == "select":
            stage_select(language, cfg, a.hf_token)
        elif s == "lexicon":
            stage_lexicon(language, cfg)
        elif s == "align":
            stage_align(language, cfg, a.num_jobs, fast=a.fast_align)
        else:
            stage_features(language, cfg)
    print("\npreprocessing stage(s) finished OK")


def main_train(language, config_path):
    ap = argparse.ArgumentParser(description="{} training".format(language))
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--max_minutes", type=float, default=None, help="stop (and save) after this many minutes")
    ap.add_argument("--total_step", type=int, default=None, help="override total_step from config.yaml")
    ap.add_argument("--plots_dir", default=None, help="where to write training/validation loss PNGs")
    ap.add_argument("--plots_only", action="store_true", help="do not train, only (re)draw the loss plots")
    a = ap.parse_args()
    cfg, configs = load_config(config_path, a.work_dir)
    from common.synth import plot_losses

    out = a.plots_dir or os.path.join(os.path.dirname(cfg["_lang_dir"]), "Results", language.capitalize())
    if a.plots_only:
        plot_losses(configs[2]["path"]["log_path"], out)
        return
    from common.train_loop import train

    try:
        train(configs, max_minutes=a.max_minutes, total_step=a.total_step)
    finally:
        plot_losses(configs[2]["path"]["log_path"], out)


def main_inference(language, config_path):
    import torch

    ap = argparse.ArgumentParser(description="{} inference".format(language))
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--checkpoint", default="best", help="best | last | path to a .pth file")
    ap.add_argument("--sentences", default=os.path.join(os.path.dirname(config_path), "test_sentences.json"))
    ap.add_argument("--out_dir", default=None, help="default: Results/<Language>")
    ap.add_argument("--speaker", default=None, help="speaker name (default: speaker with most training data)")
    ap.add_argument("--all_speakers", action="store_true", help="also synthesise with every other speaker")
    ap.add_argument("--g2p_mode", default="context", choices=["context", "lexicon"],
                    help="English only: context = g2p_en on the whole sentence, lexicon = word-by-word lookup")
    ap.add_argument("--no_normalize", action="store_true", help="skip number/time normalisation (for the ablation)")
    a = ap.parse_args()
    cfg, configs = load_config(config_path, a.work_dir)
    pre, mod, tr = configs
    from scipy.io import wavfile

    from common import synth
    from utils.model import get_vocoder
    from model import FastSpeech2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = _frontend(language)
    out_root = a.out_dir or os.path.join(os.path.dirname(cfg["_lang_dir"]), "Results", language.capitalize())
    audio_dir = os.path.join(out_root, "test_generated_audios")
    plot_dir = os.path.join(out_root, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)

    ck = a.checkpoint
    if ck in ("best", "last"):
        ck = os.path.join(tr["path"]["ckpt_path"], "{}_model_fp16.pth".format(ck))
    if not os.path.exists(ck):
        alt = os.path.join(cfg["_lang_dir"], "checkpoints", os.path.basename(ck))
        ck = alt if os.path.exists(alt) else ck
    print("checkpoint:", ck)
    model = FastSpeech2(pre, mod).to(device)
    synth.load_weights(model, ck, device)
    vocoder = get_vocoder(mod, device)

    default_name, default_id, spk_map = synth.speaker_with_most_data(pre["path"]["preprocessed_path"])
    if a.speaker:
        default_name, default_id = a.speaker, spk_map[a.speaker]
    speakers = [(default_name, default_id, True)]
    if a.all_speakers:
        speakers += [(n, i, False) for n, i in spk_map.items() if n != default_name]
    print("main speaker: {} (id {}); {} speakers in model".format(default_name, default_id, len(spk_map)))

    with open(a.sentences, encoding="utf-8") as f:
        sentences = json.load(f)["sentences"]
    lexicon = None
    if language == "english" and a.g2p_mode == "lexicon":
        lexicon = fe.read_lexicon(pre["path"]["base_lexicon"])

    results = {}
    for s in sentences:
        sid, text = s["id"], s["text"]
        if language == "english":
            phones = fe.text_to_phones(text, lexicon=lexicon, mode=a.g2p_mode, normalize=not a.no_normalize)
        else:
            phones = fe.text_to_phones(text, normalize=not a.no_normalize)
        for name, spk_id, main in speakers:
            res = synth.synthesize_one(model, vocoder, configs, phones, spk_id, device)
            if main:
                wavfile.write(os.path.join(audio_dir, sid + ".wav"), pre["preprocessing"]["audio"]["sampling_rate"], res["wav"])
                synth.save_variance_plots(res, sid, plot_dir)
                synth.save_mel_plot(res, sid, os.path.join(plot_dir, sid + "_mel.png"))
                results[sid] = {k: v for k, v in res.items() if k not in ("wav", "mel")}
                results[sid]["text"] = text
                results[sid]["speaker"] = name
                print("{}: {} phones, {:.2f} s".format(sid, len(phones), res["total_seconds"]))
            else:
                d = os.path.join(audio_dir, "extra_speakers")
                os.makedirs(d, exist_ok=True)
                wavfile.write(os.path.join(d, "{}_{}.wav".format(sid, name)),
                              pre["preprocessing"]["audio"]["sampling_rate"], res["wav"])
    with open(os.path.join(out_root, "variance_adaptor_values.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("done. audio ->", audio_dir, "| plots ->", plot_dir)
