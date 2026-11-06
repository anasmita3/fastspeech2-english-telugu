"""Diagnostic: is the weak English audio a data/generalisation problem or a pipeline problem?
For a few validation and training utterances it writes three versions:
  gt        - the real mel-spectrogram passed through the vocoder (tests the vocoder and mel settings)
  teacher   - the model run with the TRUE durations, pitch and energy (tests the acoustic model alone)
  predicted - the normal text-to-speech path (everything predicted from the text)
"""
import os, sys, json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--lang_dir", required=True)          # e.g. /content/project/English
ap.add_argument("--checkpoint", default="last")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()
ROOT = os.path.dirname(a.lang_dir.rstrip("/"))
sys.path.insert(0, ROOT)
from common.configio import load_config, setup_paths
setup_paths()
import numpy as np, torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
cfg, configs = load_config(os.path.join(a.lang_dir, "config.yaml"))
pre, mod, tr = configs
from dataset import Dataset
from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer
from utils.tools import to_device
from common import synth

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = a.checkpoint
if ck in ("best", "last"):
    ck = os.path.join(tr["path"]["ckpt_path"], "%s_model_fp16.pth" % ck)
model = FastSpeech2(pre, mod).to(dev)
synth.load_weights(model, ck, dev)
voc = get_vocoder(mod, dev)
sr = pre["preprocessing"]["audio"]["sampling_rate"]
hop = pre["preprocessing"]["stft"]["hop_length"]
os.makedirs(a.out_dir, exist_ok=True)
info = {}
bs = tr["optimizer"]["batch_size"]
for split in ("val", "train"):
    ds = Dataset(split + ".txt", pre, tr, sort=False, drop_last=False)
    loader = DataLoader(ds, batch_size=bs * 4, shuffle=False, collate_fn=ds.collate_fn)
    batch = to_device(next(iter(loader))[0], dev)
    (ids, raw, spk, texts, src_lens, max_src, mels, mel_lens, max_mel, pit, ene, dur) = batch
    with torch.no_grad():
        tf = model(*batch[2:])                      # teacher forcing: true durations/pitch/energy
    n = min(a.n, len(ids))
    for i in range(n):
        L = int(src_lens[i]); T = int(mel_lens[i])
        def wav(mel, frames):
            return vocoder_infer(mel.unsqueeze(0).transpose(1, 2), voc, mod, pre, lengths=[frames * hop])[0]
        gt = wav(mels[i, :T], T)
        te = wav(tf[1][i, :T], T)
        with torch.no_grad():
            pr = model(spk[i:i + 1], texts[i:i + 1, :L], src_lens[i:i + 1], L)
        pm = int(pr[9][0]); pw_ = wav(pr[1][0, :pm], pm)
        for tag, w in (("gt", gt), ("teacher", te), ("predicted", pw_)):
            wavfile.write(os.path.join(a.out_dir, "%s%d_%s.wav" % (split, i, tag)), sr, w)
        info["%s%d" % (split, i)] = raw[i]
        print("%s%d: %s" % (split, i, raw[i]))
json.dump(info, open(os.path.join(a.out_dir, "texts.json"), "w"), ensure_ascii=False, indent=1)
print("written to", a.out_dir)
