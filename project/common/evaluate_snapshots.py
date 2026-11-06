"""Objective checkpoint evaluation on the validation set (no listening, no ASR).

For every checkpoint it runs the normal text-to-speech path (durations, pitch and energy all
predicted from the text) on held-out validation utterances and measures:
  mel_dist    DTW-aligned mean distance between the generated and the real log-mel-spectrogram
              (how close the generated sound is to the real one; lower is better)
  dur_mae     mean absolute error of the predicted phoneme durations, in frames
  rate_ratio  generated length / real length (1.0 = same speaking rate)
  pitch_corr, energy_corr   correlation of the predicted phoneme-level pitch / energy with the real one
  pitch_rmse, energy_rmse   the same in normalised units (about 1.0 = no better than predicting the mean)
Results go to a CSV, a figure, and a printed ranking. The checkpoint with the lowest mel_dist is
the one used for the final results, instead of the validation loss.
"""
import os, sys, glob, re, argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--lang_dir", required=True)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--n", type=int, default=150, help="number of validation utterances")
a = ap.parse_args()
ROOT = os.path.dirname(a.lang_dir.rstrip("/"))
sys.path.insert(0, ROOT)
from common.configio import load_config, setup_paths
setup_paths()
import numpy as np, torch, librosa
from torch.utils.data import DataLoader
cfg, configs = load_config(os.path.join(a.lang_dir, "config.yaml"))
pre, mod, tr = configs
from dataset import Dataset
from model import FastSpeech2
from utils.tools import to_device
from common import synth

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = Dataset("val.txt", pre, tr, sort=False, drop_last=False)
loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=ds.collate_fn)
batches = []
n_seen = 0
for group in loader:
    for b in group:
        batches.append(to_device(b, dev))
        n_seen += len(b[0])
    if n_seen >= a.n:
        break

ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "snapshots", "step_*_fp16.pth")),
               key=lambda p: int(re.findall(r"step_(\d+)_", p)[0]))
labels = [("step %s" % re.findall(r"step_(\d+)_", p)[0], p) for p in ckpts]
for tag in ("best", "last"):
    p = os.path.join(a.ckpt_dir, "%s_model_fp16.pth" % tag)
    if os.path.exists(p):
        labels.append((tag, p))
print("evaluating %d checkpoints on %d validation utterances" % (len(labels), n_seen))


def corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(path):
    model = FastSpeech2(pre, mod).to(dev)
    synth.load_weights(model, path, dev)
    m = {k: [] for k in ("mel_dist", "dur_mae", "rate_ratio", "pitch_corr", "energy_corr", "pitch_rmse", "energy_rmse")}
    for batch in batches:
        (ids, raw, spk, texts, src_lens, max_src, mels, mel_lens, max_mel, pit, ene, dur) = batch
        with torch.no_grad():
            out = model(spk, texts, src_lens, max_src)
        post, p_pred, e_pred, d_rounded, pmel_lens = out[1], out[2], out[3], out[5], out[9]
        for i in range(len(ids)):
            L, T, Tp = int(src_lens[i]), int(mel_lens[i]), int(pmel_lens[i])
            g = mels[i, :T].cpu().numpy().T
            p = post[i, :Tp].float().cpu().numpy().T
            D, wp = librosa.sequence.dtw(X=g, Y=p, metric="euclidean")
            m["mel_dist"].append(float(np.mean([np.linalg.norm(g[:, x] - p[:, y]) for x, y in wp])))
            m["rate_ratio"].append(Tp / max(T, 1))
            m["dur_mae"].append(float(np.mean(np.abs(d_rounded[i, :L].float().cpu().numpy() - dur[i, :L].float().cpu().numpy()))))
            for nm, pr, tg in (("pitch", p_pred, pit), ("energy", e_pred, ene)):
                x = pr[i, :L].float().cpu().numpy(); y = tg[i, :L].float().cpu().numpy()
                m[nm + "_corr"].append(corr(x, y))
                m[nm + "_rmse"].append(float(np.sqrt(np.mean((x - y) ** 2))))
    return {k: float(np.nanmean(v)) for k, v in m.items()}


os.makedirs(a.out_dir, exist_ok=True)
rows = []
for lab, p in labels:
    r = evaluate(p)
    r["checkpoint"] = lab
    rows.append(r)
    print("%-10s mel_dist %.3f | dur_mae %.2f | rate %.2f | pitch r %.2f rmse %.2f | energy r %.2f rmse %.2f" % (
        lab, r["mel_dist"], r["dur_mae"], r["rate_ratio"], r["pitch_corr"], r["pitch_rmse"], r["energy_corr"], r["energy_rmse"]))
import pandas as pd
df = pd.DataFrame(rows)[["checkpoint", "mel_dist", "dur_mae", "rate_ratio", "pitch_corr", "pitch_rmse", "energy_corr", "energy_rmse"]]
df.to_csv(os.path.join(a.out_dir, "checkpoint_metrics.csv"), index=False)
best = df.sort_values("mel_dist").iloc[0]
print("\nRANKING by mel_dist (lower is better):")
print(df.sort_values("mel_dist")[["checkpoint", "mel_dist", "dur_mae", "pitch_corr", "energy_corr"]].to_string(index=False))
print("\n>> selected checkpoint:", best["checkpoint"])
json.dump({"selected": best["checkpoint"], "n_utterances": n_seen}, open(os.path.join(a.out_dir, "selected.json"), "w"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
snap = df[df.checkpoint.str.startswith("step")]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for axx, col, ttl in zip(ax, ("mel_dist", "dur_mae", "pitch_corr", "energy_corr"),
                         ("mel distance (lower better)", "duration error, frames (lower better)",
                          "pitch correlation (higher better)", "energy correlation (higher better)")):
    axx.plot([int(s.split()[1]) for s in snap.checkpoint], snap[col], "o-")
    axx.set_title(ttl, fontsize=9); axx.set_xlabel("training step"); axx.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "checkpoint_metrics.png"), dpi=130)
print("written to", a.out_dir)
