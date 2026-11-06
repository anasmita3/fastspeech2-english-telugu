"""
Telugu text front end for FastSpeech 2.

Two jobs:
  1. normalize_text(): turn digits / times / ordinals into Telugu words, so the
     model never sees a digit.
  2. text_to_phones(): turn Telugu script into a list of phone tokens.

Telugu spelling is almost phonetic, so a hand-written rule table is enough
(no neural G2P needed). Rules used:
  * every consonant letter carries an inherent short "a" unless it is followed
    by a vowel sign (matra) or by the virama (the "killer" sign ్).
  * consonant + virama + SAME consonant  -> geminate (long) consonant, written
    as the phone twice (e.g. పట్టు = p a tt tt u).
  * anusvara (ం) and chandrabindu (ఁ) -> "m", visarga (ః) -> "h".
  * Telugu has no schwa-deletion, so a final inherent "a" is kept.

NOTE: number words and ordinals were written from standard Telugu usage but
have NOT been checked by a native speaker. Have a Telugu speaker check
NUMBER_TEST in this file's __main__ if you can (see README).
"""
import re
import unicodedata

# ----------------------------------------------------------------- phone set
VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANT_PHONES = [
    "k", "kh", "g", "gh", "ng",
    "ch", "chh", "j", "jh", "ny",
    "tt", "tth", "dd", "ddh", "nn",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "rr", "l", "ll", "v", "sh", "ss", "s", "h",
]
# "sp" = short pause (punctuation / word gaps found by the aligner)
PHONES = VOWEL_PHONES + CONSONANT_PHONES
ALL_TOKENS = PHONES + ["sp", "spn", "sil"]

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
}
VIRAMA = "్"
NASALS = {"ం": "m", "ఁ": "m", "ః": "h"}
IGNORE = {"‌", "‍", "​", "﻿"}  # zero-width joiners etc.

TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}

# ------------------------------------------------------------ number words
_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
_TEENS = {
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది",
}
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below_100(n):
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else " " + _ONES[o])


def number_to_telugu(n):
    """0 <= n < 100000 -> Telugu words. Larger numbers are read digit by digit."""
    if n < 0 or n >= 100000:
        return " ".join(_ONES[int(d)] for d in str(abs(n)))
    if n < 100:
        return _below_100(n)
    parts = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        if thousands == 1:
            parts.append("వెయ్యి")
        else:
            parts.append(_below_100(thousands) + (" వేలు" if rest == 0 else " వేల"))
    hundreds, rest2 = divmod(rest, 100)
    if hundreds:
        if hundreds == 1:
            parts.append("వంద" if rest2 == 0 else "నూట")
        else:
            parts.append(_ONES[hundreds] + (" వందలు" if rest2 == 0 else " వందల"))
    if rest2:
        parts.append(_below_100(rest2))
    return " ".join(parts)


def ordinal_to_telugu(n):
    """12 -> పన్నెండవ. Used for '12వ'."""
    if n == 1:
        return "మొదటి"
    words = number_to_telugu(n)
    special = {"ఇరవై": "ఇరవయ్యవ", "ముప్పై": "ముప్పయ్యవ", "నలభై": "నలభయ్యవ", "యాభై": "యాభయ్యవ",
               "అరవై": "అరవయ్యవ", "డెబ్బై": "డెబ్బయ్యవ", "ఎనభై": "ఎనభయ్యవ", "తొంభై": "తొంభయ్యవ"}
    last = words.split(" ")[-1]
    if last in special:
        return " ".join(words.split(" ")[:-1] + [special[last]])
    if last.endswith("ు") or last.endswith("ి"):
        last = last[:-1]
    return " ".join(words.split(" ")[:-1] + [last + "వ"])


# ----------------------------------------------------------- normalisation
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_ORD_RE = re.compile(r"(\d+)\s?వ(?![ఀ-౿])")
_NUM_RE = re.compile(r"\d+")


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c not in IGNORE)
    text = "".join(TELUGU_DIGITS.get(c, c) for c in text)

    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        # the sentence normally continues with "గంటలకు" (at ... o'clock), so we
        # only read the two numbers: 10:45 -> "పది నలభై ఐదు"
        return number_to_telugu(h) + ("" if mi == 0 else " " + number_to_telugu(mi))

    text = _TIME_RE.sub(_time, text)
    text = _ORD_RE.sub(lambda m: ordinal_to_telugu(int(m.group(1))), text)
    text = _NUM_RE.sub(lambda m: number_to_telugu(int(m.group(0))), text)
    text = re.sub(r"[“”\"'‘’()\[\]{}<>/\\|_*#@~^`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------- G2P
_PUNCT_PAUSE = set(",.;:!?-–—।…")


def word_to_phones(word):
    """One Telugu word (no spaces, no punctuation) -> list of phone tokens."""
    chars = [c for c in word if c not in IGNORE]
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in INDEPENDENT_VOWELS:
            out += INDEPENDENT_VOWELS[c]
            i += 1
        elif c in CONSONANTS:
            ph = CONSONANTS[c]
            nxt = chars[i + 1] if i + 1 < len(chars) else None
            if nxt == VIRAMA:
                out.append(ph)  # no vowel after this consonant
                # geminate: C ్ C  -> the same consonant is emitted again by the loop
                i += 2
            elif nxt in VOWEL_SIGNS:
                out.append(ph)
                out += VOWEL_SIGNS[nxt]
                i += 2
            else:
                out += [ph, "a"]
                i += 1
        elif c in NASALS:
            out.append(NASALS[c])
            i += 1
        else:
            i += 1  # unknown character: skip it
    return out


def split_words(text):
    """normalized text -> list of ('word', w) / ('pause', None)."""
    items = []
    for tok in re.findall(r"[^\s,.;:!?\-–—।…]+|[,.;:!?\-–—।…]", text):
        if tok in _PUNCT_PAUSE:
            if items and items[-1][0] != "pause":
                items.append(("pause", None))
        else:
            items.append(("word", tok))
    return items


def text_to_phones(text, normalize=True):
    """Telugu sentence -> flat phone list; punctuation becomes 'sp'."""
    if normalize:
        text = normalize_text(text)
    phones = []
    for kind, w in split_words(text):
        if kind == "pause":
            phones.append("sp")
        else:
            phones += word_to_phones(w)
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones


def build_lexicon(texts, normalize=True):
    """Iterable of sentences -> {word: [phones]} for the aligner."""
    lex = {}
    for t in texts:
        t = normalize_text(t) if normalize else t
        for kind, w in split_words(t):
            if kind == "word" and w not in lex:
                ph = word_to_phones(w)
                if ph:
                    lex[w] = ph
    return lex


def write_lexicon(lex, path):
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted(lex):
            f.write("{}\t{}\n".format(w, " ".join(lex[w])))


NUMBER_TEST = [(0, "సున్నా"), (7, "ఏడు"), (12, "పన్నెండు"), (21, "ఇరవై ఒకటి"), (45, "నలభై ఐదు"),
               (100, "వంద"), (101, "నూట ఒకటి"), (200, "రెండు వందలు"), (2026, "రెండు వేల ఇరవై ఆరు")]

if __name__ == "__main__":
    for n, want in NUMBER_TEST:
        got = number_to_telugu(n)
        print("OK " if got == want else "BAD", n, got)
    s = "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    print(normalize_text(s))
    print(" ".join(text_to_phones(s)))
