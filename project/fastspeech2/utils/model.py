import os
import json
import zipfile

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        # [PATCH] map_location so a GPU checkpoint also loads on CPU; weights_only
        # is False because our resume checkpoints also hold optimizer state.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


_HERE = os.path.dirname(os.path.abspath(__file__))
_HIFI = os.path.join(os.path.dirname(_HERE), "hifigan")


def _unzip_if_needed(name):
    """[PATCH] the pretrained HiFi-GAN weights ship zipped; unzip on first use."""
    target = os.path.join(_HIFI, name)
    if not os.path.exists(target):
        with zipfile.ZipFile(target + ".zip") as z:
            z.extractall(_HIFI)
    return target


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "HiFi-GAN":
        with open(os.path.join(_HIFI, "config.json"), "r") as f:
            hconfig = json.load(f)
        hconfig = hifigan.AttrDict(hconfig)
        vocoder = hifigan.Generator(hconfig)
        fname = {"LJSpeech": "generator_LJSpeech.pth.tar", "universal": "generator_universal.pth.tar"}[speaker]
        ckpt = torch.load(_unzip_if_needed(fname), map_location=device, weights_only=False)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    else:
        raise ValueError("only HiFi-GAN is supported in this project")

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
