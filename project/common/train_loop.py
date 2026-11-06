"""
Training loop for FastSpeech 2 (re-uses the model / loss / dataset of the original repo).

Differences from the original train.py (all on purpose, see README "Changes"):
  * resumable: a full checkpoint (resume.pth.tar) is written every `save_step`
    steps; running the same command again continues from it (Colab disconnects!).
  * losses go to plain CSV files (train_loss.csv / val_loss.csv) so the loss
    plots can be drawn without TensorBoard. Training loss is the average over
    the last `log_step` steps (less noisy than a single batch).
  * validation loss is computed on the whole validation set every `val_step`.
  * the best model (lowest validation loss) is kept as best_model_fp16.pth,
    the last one as last_model_fp16.pth (weights only, half precision, ~70 MB so
    that they fit on GitHub without LFS).
  * --max_minutes stops cleanly after a time budget (also saves).
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

LOSS_NAMES = ["total", "mel", "postnet", "pitch", "energy", "duration"]


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def run_validation(model, configs, device, Loss, Dataset):
    pre, mod, tr = configs
    dataset = Dataset("val.txt", pre, tr, sort=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=tr["optimizer"]["batch_size"], shuffle=False,
                        collate_fn=dataset.collate_fn)
    from utils.tools import to_device

    sums = [0.0] * 6
    model.eval()
    with torch.no_grad():
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                out = model(*(batch[2:]))
                losses = Loss(batch, out)
                for i in range(6):
                    sums[i] += losses[i].item() * len(batch[0])
    model.train()
    return [s / len(dataset) for s in sums]


def train(configs, work_log_dir=None, max_minutes=None, total_step=None, device=None, quiet=False):
    from dataset import Dataset
    from model import FastSpeech2Loss
    from utils.model import get_model
    from utils.tools import to_device

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre, mod, tr = configs
    ckpt_dir = tr["path"]["ckpt_path"]
    log_dir = work_log_dir or tr["path"]["log_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, "resume.pth.tar")

    step_cfg = tr["step"]
    total_step = total_step or step_cfg["total_step"]
    log_step, val_step, save_step = step_cfg["log_step"], step_cfg["val_step"], step_cfg["save_step"]
    grad_acc = tr["optimizer"]["grad_acc_step"]
    clip = tr["optimizer"]["grad_clip_thresh"]
    bs = tr["optimizer"]["batch_size"]

    dataset = Dataset("train.txt", pre, tr, sort=True, drop_last=True)
    group = 4
    assert bs * group < len(dataset), "training set too small ({} utterances)".format(len(dataset))
    loader = DataLoader(dataset, batch_size=bs * group, shuffle=True, collate_fn=dataset.collate_fn,
                        num_workers=2, persistent_workers=True)

    class _Args:
        restore_step = 0

    model, optimizer = get_model(_Args, configs, device, train=True)
    Loss = FastSpeech2Loss(pre, mod).to(device)
    print("parameters: {:.1f} M, device: {}".format(sum(p.numel() for p in model.parameters()) / 1e6, device))

    step, best_val = 0, float("inf")
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        optimizer.current_step = ck["step"]
        step, best_val = ck["step"], ck.get("best_val", float("inf"))
        print("RESUMED from step {} (best val so far {:.4f})".format(step, best_val))
    else:
        print("starting from scratch")

    snap_every = configs[2]["step"].get("snapshot_step", 0)

    def save_all(tag=""):
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer._optimizer.state_dict(),
                      "step": step, "best_val": best_val}, resume_path)
        _atomic_save({k: v.half() for k, v in model.state_dict().items()},
                     os.path.join(ckpt_dir, "last_model_fp16.pth"))
        if snap_every and step % snap_every == 0:  # v2: keep a copy per snapshot, never overwritten
            os.makedirs(os.path.join(ckpt_dir, "snapshots"), exist_ok=True)
            _atomic_save({k: v.half() for k, v in model.state_dict().items()},
                         os.path.join(ckpt_dir, "snapshots", "step_{}_fp16.pth".format(step)))
        if not quiet:
            print("saved checkpoint at step {} {}".format(step, tag))

    header_tr = ["step"] + LOSS_NAMES + ["seconds"]
    t0 = time.time()
    window = [0.0] * 6
    n_window = 0
    model.train()
    done = step >= total_step
    while not done:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                out = model(*(batch[2:]))
                losses = Loss(batch, out)
                (losses[0] / grad_acc).backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                step += 1
                for i in range(6):
                    window[i] += losses[i].item()
                n_window += 1

                if step % log_step == 0:
                    row = [step] + [round(w / n_window, 5) for w in window] + [round(time.time() - t0)]
                    _append_csv(os.path.join(log_dir, "train_loss.csv"), header_tr, row)
                    if not quiet:
                        print("step {}/{}  loss {:.3f} (mel {:.3f} postnet {:.3f} pitch {:.3f} energy {:.3f} dur {:.3f})".format(
                            step, total_step, *[w / n_window for w in window]), flush=True)
                    window, n_window = [0.0] * 6, 0

                if step % val_step == 0 or step == total_step:
                    vals = run_validation(model, configs, device, Loss, Dataset)
                    _append_csv(os.path.join(log_dir, "val_loss.csv"), header_tr,
                                [step] + [round(v, 5) for v in vals] + [round(time.time() - t0)])
                    print("VALIDATION step {}  total {:.3f} (mel {:.3f} postnet {:.3f} pitch {:.3f} energy {:.3f} dur {:.3f})".format(
                        step, *vals), flush=True)
                    if vals[0] < best_val:
                        best_val = vals[0]
                        _atomic_save({k: v.half() for k, v in model.state_dict().items()},
                                     os.path.join(ckpt_dir, "best_model_fp16.pth"))
                        print("  new best validation loss -> best_model_fp16.pth")

                if step % save_step == 0:
                    save_all()

                over_time = max_minutes is not None and (time.time() - t0) / 60 > max_minutes
                if step >= total_step or over_time:
                    save_all("(final)" if step >= total_step else "(time budget reached - run again to continue)")
                    done = True
                    break
            if done:
                break
    print("finished at step", step)
    return step
