"""
English text front end.

normalize_text():  digits / times / years -> words, lower-case, tidy punctuation.
text_to_phones():  sentence -> ARPAbet phones (with stress digits) + 'sp' for pauses.

Two ways to get phones at inference time (compare them in the report!):
  mode="context"  : run g2p_en on the WHOLE sentence. g2p_en looks at the part of
                    speech, so heteronyms such as "conduct" (noun) vs "conduct"
                    (verb) can get different stress.
  mode="lexicon"  : plain dictionary lookup word by word (what the original
                    FastSpeech 2 repo does). Same spelling -> always the same
                    phones, so noun/verb "conduct" sound identical.
Words missing from the lexicon fall back to g2p_en in both modes.
"""
import re

import inflect

_inf = inflect.engine()

_MONTHS = ("january february march april may june july august september october "
           "november december").split()
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?")
_YEAR_RE = re.compile(r"(?<!\d)(1[1-9]\d\d|20\d\d)(?!\d)")
_NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _num(n):
    return _inf.number_to_words(n, andword="", comma="").replace("-", " ").replace(",", "")


def _year_words(y):
    if 2000 <= y <= 2009:
        return _num(y)
    a, b = divmod(y, 100)
    if b == 0:
        return _num(a) + " hundred"
    return _num(a) + " " + ("oh " + _num(b) if b < 10 else _num(b))


def _time_words(m):
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    out = _num(h)
    if mi == 0:
        out += " o'clock" if not ap else ""
    elif mi < 10:
        out += " oh " + _num(mi)
    else:
        out += " " + _num(mi)
    if ap:
        out += " " + " ".join(ap.replace(".", "").lower())  # "AM" -> "a m"
    return out


def normalize_text(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _TIME_RE.sub(_time_words, text)
    text = _YEAR_RE.sub(lambda m: _year_words(int(m.group(1))), text)

    def _generic(m):
        s = m.group(0).replace(",", "")
        if "." in s:
            whole, frac = s.split(".")
            return _num(int(whole)) + " point " + " ".join(_num(int(d)) for d in frac)
        return _num(int(s))

    text = _NUM_RE.sub(_generic, text)
    text = text.lower()
    text = re.sub(r"[^a-z' ,.;:!?\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_training(text):
    """Same as normalize_text, used on transcripts so train and test text match."""
    text = re.sub(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)", " ", text)  # [noise], <unk>, (laughs)
    return normalize_text(text)


# ------------------------------------------------------------------ lexicon
def read_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) > 1 and parts[0].lower() not in lex:
                lex[parts[0].lower()] = parts[1:]
    return lex


_ARPA = re.compile(r"^[A-Z]{1,2}[0-2]?$")  # real ARPAbet phones only
_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk

        for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from g2p_en import G2p

        _g2p = G2p()
    return _g2p


_PAUSE = re.compile(r"^[,;:.!?\-]+$")
_ARPA = re.compile(r"^[A-Z]{1,2}[0-2]?$")  # real ARPAbet phones; g2p_en can also return "'" etc.


def text_to_phones(text, lexicon=None, mode="context", normalize=True):
    """-> list of phone tokens (ARPAbet + 'sp')."""
    if normalize:
        text = normalize_text(text)
    g2p = _get_g2p()
    phones = []
    if mode == "context":
        for p in g2p(text):
            if p == " ":
                continue
            if _PAUSE.match(p):
                phones.append("sp")
            elif _ARPA.match(p):
                phones.append(p)
    elif mode == "lexicon":
        assert lexicon is not None, "lexicon mode needs a lexicon"
        for w in re.split(r"([,;:.!?\-]|\s+)", text):
            if not w or w.isspace():
                continue
            if _PAUSE.match(w):
                phones.append("sp")
            elif w in lexicon:
                phones += lexicon[w]
            else:
                phones += [p for p in g2p(w) if _ARPA.match(p)]
    else:
        raise ValueError(mode)
    # collapse repeated pauses, drop leading/trailing pauses
    out = []
    for p in phones:
        if p == "sp" and (not out or out[-1] == "sp"):
            continue
        out.append(p)
    while out and out[-1] == "sp":
        out.pop()
    return out


def build_lexicon_for_corpus(texts, base_lexicon):
    """Words of the (already normalised) transcripts -> {word: phones}.
    Words absent from base_lexicon get a g2p_en pronunciation."""
    g2p = _get_g2p()
    words = set()
    for t in texts:
        words.update(w for w in re.findall(r"[a-z']+", t))
    lex = {}
    for w in sorted(words):
        if w in base_lexicon:
            lex[w] = base_lexicon[w]
        else:
            ph = [p for p in g2p(w) if _ARPA.match(p)]
            if ph:
                lex[w] = ph
    return lex


def write_lexicon(lex, path):
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted(lex):
            f.write("{}\t{}\n".format(w, " ".join(lex[w])))


if __name__ == "__main__":
    S = {
        "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
        "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
        "E3": "The rhythm of the orchestra changed unexpectedly.",
        "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
    }
    for k, v in S.items():
        print(k, "|", normalize_text(v))
        print("   ", " ".join(text_to_phones(v)))
