"""Normalize raw Vietnamese text into the input expected by CharsiuG2P `infer_sentence`.

See g2p/PHONEME_FORMAT.md section 7. Each step is recorded so a debug UI can show where the text changed.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from tts_amis.numbers import to_words

PUNCTUATION = ",.!?;:"

# Tone mark on the first vowel of open oa/oe/uy ("hóa", "thủy") moves to the second vowel ("hoá", "thuỷ"),
# the spelling used by most vie-n.tsv keys.
_TONED = {"a": "àáảãạ", "e": "èéẻẽẹ", "y": "ỳýỷỹỵ", "o": "òóỏõọ", "u": "ùúủũụ"}
_TONE_MOVES = {
    _TONED[first][i] + second: first + _TONED[second][i]
    for first, seconds in (("o", "ae"), ("u", "y"))
    for second in seconds
    for i in range(5)
}
_TONE_RE = re.compile("(" + "|".join(_TONE_MOVES) + r")(?!\w)")

# NFD components allowed in a letter: Latin a-z, đ, and Vietnamese combining marks.
_COMBINING_OK = {"̀", "́", "̃", "̉", "̣", "̂", "̆", "̛"}
_KEEP_CHARS = set(PUNCTUATION) | {"-", " "}


@dataclass
class Normalized:
    text: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _is_letter(ch: str) -> bool:
    if ch == "đ":
        return True
    parts = unicodedata.normalize("NFD", ch)
    return "a" <= parts[0] <= "z" and all(p in _COMBINING_OK for p in parts[1:])


def _numbers_to_words(text: str) -> str:
    def repl(m: re.Match) -> str:
        words = to_words(int(m.group()))
        before = text[m.start() - 1] if m.start() else " "
        after = text[m.end()] if m.end() < len(text) else " "
        if before == "-" or after == "-":
            return words.replace(" ", "-")
        return (" " if before.isalpha() else "") + words + (" " if after.isalpha() else "")

    return re.sub(r"\d+", repl, text)


def normalize_with_info(raw: str) -> Normalized:
    """Run every step and keep the intermediate strings. Raises ValueError when no letters remain."""
    steps: list[tuple[str, str]] = []

    t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()
    steps.append(("nfc_whitespace", t))
    t = t.lower()
    steps.append(("lower", t))
    t = _TONE_RE.sub(lambda m: _TONE_MOVES[m.group()], t)
    steps.append(("tone_placement", t))
    t = _numbers_to_words(t)
    steps.append(("numbers", t))

    dropped = sorted({ch for ch in t if ch not in _KEEP_CHARS and not _is_letter(ch)})
    t = "".join(ch if ch in _KEEP_CHARS or _is_letter(ch) else " " for ch in t)
    steps.append(("drop_unsupported", t))

    t = re.sub(rf"\s*([{re.escape(PUNCTUATION)}])\s*", r" \1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    steps.append(("split_punctuation", t))

    if not any(_is_letter(ch) for ch in t):
        raise ValueError(f"no readable letters in input: {raw!r}")
    return Normalized(text=t, steps=steps, dropped=dropped)


def normalize(raw: str) -> str:
    return normalize_with_info(raw).text
