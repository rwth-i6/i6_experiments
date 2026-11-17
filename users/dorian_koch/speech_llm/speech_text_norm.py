"""Text normalisation for the TTS round-trip check (``ChatterboxInferenceV2``'s regeneration pass).

The text we asked the TTS to say and the ASR transcript of what it said are brought to one spoken form, so that
writing differences do not count as errors: digits vs number words ("3" / "drei", "1998" / "neunzehnhundert-
achtundneunzig", "25" / "二十五"), symbols ("%" / "Prozent"), case, punctuation and hyphenation. Only real
differences in what was said remain.

* English: Whisper's ``EnglishTextNormalizer`` (``whisper_normalizer``; number words to digits, spelling,
  contractions), as ``text_scoring.norm_words`` scores English elsewhere; "cannot" = "can not".
* Other space-delimited languages: digits to words with ``num2words`` (a four-digit 1100-2099 as a year), then case,
  punctuation and hyphens removed. Russian: ё -> е.
* Chinese / Japanese / Cantonese: Arabic numerals to Chinese numerals (``cn2an``; "40%" -> 百分之四十), 两 -> 二, punctuation and spaces
  removed; scored per character (Latin words stay one token).

:func:`error_rate` gives WER and CER (both CER for the character languages). Needs ``num2words``, ``cn2an`` and ``whisper_normalizer``:
it runs inside ``speech_annotate_worker.py`` under the annotate venv.
"""

from __future__ import annotations

import re
import unicodedata

CHAR_LANGS = frozenset({"zh", "ja", "yue"})  # as chatterbox_v2_core.CHAR_LANGS

#: symbols spoken as words, per language (applied before punctuation is dropped)
SYMBOLS = {
    "de": {"%": " prozent ", "&": " und ", "+": " plus ", "€": " euro ", "$": " dollar ", "°": " grad "},
    "fr": {"%": " pour cent ", "&": " et ", "+": " plus ", "€": " euros ", "$": " dollars ", "°": " degrés "},
    "es": {"%": " por ciento ", "&": " y ", "+": " más ", "€": " euros ", "$": " dólares ", "°": " grados "},
    "ru": {"%": " процентов ", "&": " и ", "+": " плюс ", "€": " евро ", "$": " долларов ", "°": " градусов "},
    "zh": {"%": "百分之", "&": "和", "+": "加", "€": "欧元", "$": "美元", "°": "度"},
}

_EN = None


def _english(text: str) -> list[str]:
    global _EN
    if _EN is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _EN = EnglishTextNormalizer()
    return _EN(text).split()


def _num_words(text: str, lang: str) -> str:
    from num2words import num2words

    def year_or_int(m):
        s = m.group(0)
        n = int(s)
        try:
            if len(s) == 4 and 1100 <= n <= 2099:
                return f" {num2words(n, lang=lang, to='year')} "
            return f" {num2words(n, lang=lang)} "
        except NotImplementedError:
            return f" {num2words(n, lang='en')} "

    def decimal(m):
        a, b = m.group(1), m.group(2)
        try:
            return (
                f" {num2words(int(a), lang=lang)} {_comma(lang)} {' '.join(num2words(int(d), lang=lang) for d in b)} "
            )
        except NotImplementedError:
            return m.group(0)

    # thousands separators (1.000.000 / 1,000 / 1 000) only in exact groups of three
    text = re.sub(r"(?<![\d.,])\d{1,3}(?:[.,   ]\d{3})+(?![\d]|[.,]\d)", lambda m: re.sub(r"\D", "", m.group(0)), text)
    text = re.sub(r"(\d+)[.,](\d+)", decimal, text)
    return re.sub(r"\d+", year_or_int, text)


def _comma(lang: str) -> str:
    return {"de": "komma", "fr": "virgule", "es": "coma", "ru": "запятая", "it": "virgola"}.get(lang, "point")


def _drop_punct(text: str) -> str:
    return "".join(" " if unicodedata.category(c)[0] in "PSZ" or c in "-‐‑–—" else c for c in text)


def normalize(text: str, lang: str) -> list[str]:
    """Tokens of ``text`` in its spoken form: words, or characters (and whole Latin words) for ``CHAR_LANGS``."""
    text = unicodedata.normalize("NFKC", text)
    if lang == "en":
        return " ".join(_english(text)).replace("cannot", "can not").split()
    if lang in CHAR_LANGS:  # "40%" is read 百分之四十: the number moves behind the word
        import cn2an

        text = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", lambda m: "百分之" + cn2an.an2cn(m.group(1)), text)
    for s, w in SYMBOLS.get(lang, {}).items():
        text = text.replace(s, w)
    if lang in CHAR_LANGS:
        import cn2an

        text = re.sub(r"\d+(?:\.\d+)?", lambda m: cn2an.an2cn(m.group(0)), text)
        text = _drop_punct(text.casefold()).replace("两", "二")
        return re.findall(r"[a-z0-9']+|\S", text)
    text = _num_words(text, lang).casefold()
    if lang == "de":  # "(ein)hundert", "(ein)tausend": both are said
        text = re.sub(r"\bein(hundert|tausend)", r"\1", text)
    if lang == "ru":
        text = text.replace("ё", "е")
    return _drop_punct(text).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _rate(r: list, h: list) -> float:
    return edit_distance(r, h) / len(r) if r else float(bool(h))


def error_rate(ref: str, hyp: str, lang: str) -> dict:
    """``{"err", "wer", "cer", "n_ref", "ref_norm", "hyp_norm"}`` after :func:`normalize`: ``wer`` = token edits /
    reference tokens (for ``CHAR_LANGS`` the tokens are characters, so it is a CER), ``cer`` = character edits /
    reference characters of the normalised text (spaces between words count once; none for ``CHAR_LANGS``), ``err`` =
    ``wer`` (the old field). An empty reference scores 0 if the transcript is empty too, else 1."""
    r, h = normalize(ref, lang), normalize(hyp, lang)
    sep = "" if lang in CHAR_LANGS else " "
    wer = _rate(r, h)
    cer = _rate(list(sep.join(r)), list(sep.join(h)))
    return {"err": float(wer), "wer": float(wer), "cer": float(cer), "n_ref": len(r), "ref_norm": " ".join(r),
            "hyp_norm": " ".join(h)}  # fmt: skip
