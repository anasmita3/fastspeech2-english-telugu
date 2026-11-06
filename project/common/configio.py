"""Load a language's config.yaml and expand its {work}/{root}/{lang} placeholders."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
CORE = os.path.join(ROOT, "fastspeech2")


def setup_paths():
    """Make `import model`, `import text`, ... (the FastSpeech 2 code) work."""
    import sys

    for p in (CORE, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _expand(obj, subs):
    if isinstance(obj, dict):
        return {k: _expand(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v, subs) for v in obj]
    if isinstance(obj, str):
        for k, v in subs.items():
            obj = obj.replace("{" + k + "}", v)
    return obj


def load_config(config_path, work_dir=None):
    """Returns (cfg, (preprocess_config, model_config, train_config))."""
    config_path = os.path.abspath(config_path)
    lang_dir = os.path.dirname(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    work = os.path.abspath(work_dir or os.environ.get("FS2_WORK") or os.path.join(lang_dir, "work"))
    subs = {"work": work, "root": ROOT, "lang": lang_dir, "core": CORE}
    cfg = _expand(cfg, subs)
    cfg["_work"] = work
    cfg["_lang_dir"] = lang_dir
    for d in (cfg["preprocess"]["path"]["raw_path"], cfg["preprocess"]["path"]["preprocessed_path"]):
        os.makedirs(d, exist_ok=True)
    for p in cfg["train"]["path"].values():
        os.makedirs(p, exist_ok=True)
    return cfg, (cfg["preprocess"], cfg["model"], cfg["train"])
