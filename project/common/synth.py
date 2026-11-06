"""
Inference helpers: text -> phones -> FastSpeech 2 -> HiFi-GAN -> wav, and the
variance-adaptor plots (predicted duration, pitch/F0, energy for every phone).
"""
import json
import os

import numpy as np
import torch

matplotlib_ready = False


def _plt():
    global matplotlib_ready
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib_ready = True
    return plt


def load_weights(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model" in ck and not isinstance(ck["model"], torch.Tensor):
        ck = ck["model"]
    model.load_state_dict({k: v.float() for k, v in ck.items()})
    model.eval()
    return model


def speaker_with_most_data(preprocessed_path):
    """Default voice for the test sentences: the speaker with most training audio."""
    from collections import Counter

    with open(os.path.join(preprocessed_path, "speakers.json")) as f:
        spk_map = json.load(f)
    cnt = Counter()
    with open(os.path.join(preprocessed_path, "train.txt"), encoding="utf-8") as f:
        for line in f:
            cnt[line.split("|")[1]] += 1
    best = cnt.most_common(1)[0][0]
    return best, spk_map[best], spk_map


@torch.no_grad()
def synthesize_one(model, vocoder, configs, phones, speaker_id, device):
    """phones: list of tokens -> dict with wav (int16) and per-phone predictions."""
    from text import text_to_sequence
    from utils.model import vocoder_infer

    pre, mod, tr = configs
    seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", pre["preprocessing"]["text"]["text_cleaners"]))
    if len(seq) != len(phones):
        raise ValueError("some phones are not in the model's symbol table: {}".format(phones))
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(seq)]).to(device)
    speakers = torch.tensor([speaker_id]).long().to(device)
    out = model(speakers, texts, src_lens, len(seq))
    (mel, postnet_mel, p_pred, e_pred, log_d, d_round, src_masks, mel_masks, _, mel_lens) = out
    mel_len = int(mel_lens[0])
    wav = vocoder_infer(postnet_mel.transpose(1, 2), vocoder, mod, pre,
                        lengths=[mel_len * pre["preprocessing"]["stft"]["hop_length"]])[0]

    with open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_min, p_max, p_mean, p_std = stats["pitch"]
    e_min, e_max, e_mean, e_std = stats["energy"]
    hop, sr = pre["preprocessing"]["stft"]["hop_length"], pre["preprocessing"]["audio"]["sampling_rate"]
    frames = d_round[0].cpu().numpy().astype(int)
    pitch = p_pred[0].cpu().numpy()
    energy = e_pred[0].cpu().numpy()
    return {
        "wav": wav,
        "phones": list(phones),
        "duration_frames": frames.tolist(),
        "duration_ms": (frames * hop / sr * 1000).tolist(),
        "pitch_hz": (pitch * p_std + p_mean).tolist() if pre["preprocessing"]["pitch"]["normalization"] else pitch.tolist(),
        "pitch_normalised": pitch.tolist(),
        "energy": (energy * e_std + e_mean).tolist() if pre["preprocessing"]["energy"]["normalization"] else energy.tolist(),
        "energy_normalised": energy.tolist(),
        "mel": postnet_mel[0, :mel_len].cpu().numpy(),
        "total_seconds": mel_len * hop / sr,
    }


def _bar_plot(path, values, phones, title, ylabel, color, silence_mask=None):
    plt = _plt()
    n = len(values)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * n), 3.6))
    cols = [("lightgray" if (silence_mask is not None and silence_mask[i]) else color) for i in range(n)]
    ax.bar(range(n), values, color=cols)
    ax.set_xticks(range(n))
    ax.set_xticklabels(phones, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _line_plot(path, values, phones, title, ylabel, color):
    plt = _plt()
    n = len(values)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * n), 3.6))
    ax.plot(range(n), values, marker="o", color=color, linewidth=1.5, markersize=4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(phones, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_variance_plots(res, sent_id, plot_root, label=""):
    """duration/, pitch/ and energy/ sub-folders, one PNG per sentence."""
    phones = res["phones"]
    sil = np.array([p in ("sp", "spn", "sil") for p in phones])
    tag = "{}{}".format(sent_id, (" - " + label) if label else "")
    for sub in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(plot_root, sub), exist_ok=True)
    _bar_plot(os.path.join(plot_root, "duration", sent_id + ".png"), res["duration_ms"], phones,
              "{}: predicted duration per phone (total {:.2f} s)".format(tag, res["total_seconds"]),
              "duration (ms)", "tab:blue", sil)
    pitch = np.array(res["pitch_hz"])
    pitch_plot = np.where(sil, np.nan, pitch)  # pauses have no pitch
    _line_plot(os.path.join(plot_root, "pitch", sent_id + ".png"), pitch_plot, phones,
               "{}: predicted pitch (F0) per phone".format(tag), "F0 (Hz)", "tab:red")
    energy = np.array(res["energy"])
    _bar_plot(os.path.join(plot_root, "energy", sent_id + ".png"), energy, phones,
              "{}: predicted energy per phone".format(tag), "energy (mel-frame L2 norm)", "tab:purple", sil)


def save_mel_plot(res, sent_id, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.imshow(res["mel"].T, origin="lower", aspect="auto")
    ax.set_title("{}: generated mel-spectrogram".format(sent_id), fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_losses(log_dir, out_dir, title_prefix=""):
    """training_loss.png and validation_loss.png from the CSV logs."""
    import csv

    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    for kind, fname, png in (("Training", "train_loss.csv", "training_loss.png"),
                             ("Validation", "val_loss.csv", "validation_loss.png")):
        path = os.path.join(log_dir, fname)
        if not os.path.exists(path):
            print("no", path)
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(steps, [float(r["total"]) for r in rows], color="black", linewidth=1.8)
        axes[0].set_title("{}{} loss (total)".format(title_prefix, kind))
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        if kind == "Validation":
            b = int(np.argmin([float(r["total"]) for r in rows]))
            axes[0].axvline(steps[b], color="green", linestyle="--", label="best: step {}".format(steps[b]))
            axes[0].legend()
        for name in ("mel", "postnet", "pitch", "energy", "duration"):
            axes[1].plot(steps, [float(r[name]) for r in rows], label=name)
        axes[1].set_title("{}{} loss parts".format(title_prefix, kind))
        axes[1].set_xlabel("step")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, png), dpi=130)
        plt.close(fig)
        print("wrote", os.path.join(out_dir, png))
