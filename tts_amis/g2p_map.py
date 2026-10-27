"""Map one syllable's output from vPhon, viphoneme or espeak-ng onto CharsiuG2P vie-n tokens (the XPhoneBERT vocab).

Only notation is mapped (tone numbers -> Chao letters, vowel length -> a/ă, ɤ/ɤ̆, consonant symbols). Allophonic
choices each tool makes (e.g. "khách" as `x a c` instead of Charsiu's `x ɛ k`) are kept, so the comparison stays honest.
Symbols with no mapping are passed through and show up as out-of-vocab in validation.
"""

import unicodedata
from collections.abc import Callable

from tts_amis.phonemes import NANG

GLOTTAL = "ʔ"
STOP_CODAS = {"p", "t", "k", "c", "tʃ", "k͡p"}
SAC_OPEN, SAC_CHECKED = "˨˦", "˦˥"

# vPhon -d n -c -n (Chao numbers, no superscripts).
VPHON_TONES = {
    "33": ["˧˧"],
    "32": ["˧˨"],
    "24": [SAC_OPEN],
    "45": [SAC_CHECKED],
    "312": ["˧˩˨"],
    "3g5": ["˧ˀ˥"],
    "21g": [NANG, GLOTTAL],
    "21": [NANG],
}
VPHON_PHONES = {
    "aː": "a", "a": "ă", "əː": "ɤ", "ə": "ɤ̆", "iː": "i", "eː": "e", "ɛː": "ɛ", "oː": "o", "ɔː": "ɔ", "uː": "u",
    "ɨː": "ɯ", "ɨ": "ɯ", "ɨə": "ɯə", "tɕ": "c", "th": "tʰ", "ʈ": "c", "ʂ": "s", "ʐ": "z",
}  # fmt: skip

# viphoneme (Pham 2006 tone numbers 1-6, northern dialect).
VIPHONEME_PHONES = {"ʂ": "s", "ʈ": "c", "ʐ": "z", "tʃ": "c"}

# espeak-ng "vi" via phonemizer: tone marks glued to the vowel; ngang is unmarked.
ESPEAK_TONE_MARKS = {"2": "huyen", "ɜ": "sac", "4": "hoi", "5": "nga", "6": "nang"}
ESPEAK_PHONES = {
    "t̪": "t", "tʃ": "c", "aː": "a", "a": "ă", "əː": "ɤ", "ə": "ɤ̆", "y": "ɯ", "iɛ": "iə",
}  # fmt: skip

_NAMED_TONES = {"ngang": "˧˧", "huyen": "˧˨", "hoi": "˧˩˨", "nga": "˧ˀ˥"}
DIPHTHONGS = {"iə", "uə", "ɯə"}


def _nfd(fn: Callable[[str], list[str]]) -> Callable[[str], list[str]]:
    """XPhoneBERT's vocab is NFD (`ă` = a + U+0306); precomposed forms in the tables above would be out of vocab."""

    def wrapper(raw: str) -> list[str]:
        return [unicodedata.normalize("NFD", tok) for tok in fn(raw)]

    wrapper.__doc__ = fn.__doc__
    return wrapper


def _split_diphthongs(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        out += list(tok) if tok in DIPHTHONGS else [tok]
    return out


def _tone_tokens(name: str, phones: list[str]) -> list[str]:
    """Charsiu tone tokens for a tone name; sắc and nặng depend on whether the syllable ends in a stop."""
    checked = bool(phones) and phones[-1] in STOP_CODAS
    if name == "sac":
        return [SAC_CHECKED if checked else SAC_OPEN]
    if name == "nang":
        return [NANG] if checked else [NANG, GLOTTAL]
    return [_NAMED_TONES[name]]


@_nfd
def from_vphon(raw: str) -> list[str]:
    """`.t.w.aː.n.24.` -> `t w a n ˨˦`."""
    parts = [p for p in raw.split(".") if p]
    if not parts:
        return []
    *phones, tone = parts
    mapped = _split_diphthongs([VPHON_PHONES.get(p, p) for p in phones])
    return mapped + VPHON_TONES.get(tone, [tone])


@_nfd
def from_viphoneme(raw: str) -> list[str]:
    """`.tw.ʷa.n.5.` -> `t w a n ˨˦`."""
    parts = [p for p in raw.split(".") if p]
    if not parts:
        return []
    *phones, tone = parts
    mapped: list[str] = []
    for i, p in enumerate(phones):
        p = p.lstrip("ʷ")
        if i == 0 and p == "j":  # onset gi/d; a later j is the off-glide
            p = "z"
        if len(p) > 1 and p.endswith("w") and p != "w":  # onset + labial glide in one symbol: tw, kw, tʰw
            mapped += [VIPHONEME_PHONES.get(p[:-1], p[:-1]), "w"]
        elif p:
            mapped.append(VIPHONEME_PHONES.get(p, p))
    mapped = _split_diphthongs(mapped)
    names = {"1": "ngang", "2": "huyen", "3": "nga", "4": "hoi", "5": "sac", "6": "nang"}
    return mapped + (_tone_tokens(names[tone], mapped) if tone in names else [tone])


@_nfd
def from_espeak(raw: str) -> list[str]:
    """`t̪ w aːɜ n` -> `t w a n ˨˦`."""
    tone = "ngang"
    mapped: list[str] = []
    for p in raw.split():
        if p[-1] in ESPEAK_TONE_MARKS and len(p) > 1:
            tone = ESPEAK_TONE_MARKS[p[-1]]
            p = p[:-1]
        mapped.append(ESPEAK_PHONES.get(p, p))
    mapped = _split_diphthongs(mapped)
    return mapped + _tone_tokens(tone, mapped) if mapped else []
