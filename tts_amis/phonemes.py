"""Check a CharsiuG2P phoneme string against the invariants in g2p/PHONEME_FORMAT.md section 7."""

import unicodedata
from dataclasses import dataclass, field

WORD_BOUNDARY = "▁"
SYLLABLE_JOIN = "-"
GLOTTAL = "ʔ"
NANG = "˨ˀ˩"
TONES = {"˧˧", "˧˨", "˨˦", "˦˥", "˧˩˨", "˧ˀ˥", NANG}
# Chao tone letters used by CharsiuG2P vie-n, named by the Vietnamese tone they encode.
TONE_NAMES = {
    "˧˧": "ngang",
    "˧˨": "huyền",
    "˨˦": "sắc",
    "˦˥": "sắc (p/t/c/ch)",
    "˧˩˨": "hỏi",
    "˧ˀ˥": "ngã",
    NANG: "nặng",
}
PUNCTUATION_TOKENS = set(",.!?;:-")
STRESS_MARKS = {"ˈ", "ˌ", "ː"}
VOWELS = {"a", "ă", "ɤ̆", "e", "ɛ", "i", "o", "ɔ", "u", "ɯ", "ɤ", "ə"}
GLIDES = {"w", "j"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.warnings


def split_words(phonemes: str) -> list[list[str]]:
    """Token lists between `▁` boundaries; one entry per input word or punctuation mark."""
    words, current = [], []
    for tok in phonemes.split():
        if tok == WORD_BOUNDARY:
            words.append(current)
            current = []
        else:
            current.append(tok)
    words.append(current)
    return words


def split_syllables(word: list[str]) -> list[list[str]]:
    """Split a word at `-` tokens. A punctuation-only word is returned as is."""
    if len(word) == 1 and word[0] in PUNCTUATION_TOKENS:
        return [word]
    parts, current = [], []
    for tok in word:
        if tok == SYLLABLE_JOIN:
            parts.append(current)
            current = []
        else:
            current.append(tok)
    parts.append(current)
    return parts


def tone_names(phonemes: str) -> list[str]:
    """Vietnamese names of the tone tokens in order; unknown tone-like tokens such as a lone `˧` are shown as is."""
    return [TONE_NAMES.get(tok, tok) for tok in phonemes.split() if tok in TONE_NAMES or set(tok) <= set("˥˦˧˨˩ˀ")]


def _check_part(part: list[str]) -> str | None:
    """Return a problem description for one hyphen-separated part, or None if it is one well-formed syllable."""
    if len(part) == 1 and part[0] in PUNCTUATION_TOKENS:
        return None
    if not part:
        return "empty syllable"
    tone_positions = [i for i, tok in enumerate(part) if tok in TONES]
    end = len(part) - 1
    if part[end] == GLOTTAL:
        end -= 1
    if not tone_positions:
        return "no tone token"
    if len(tone_positions) > 1:
        return f"{len(tone_positions)} tone tokens in one syllable (glued multi-syllable word?)"
    if tone_positions[0] != end:
        return "tone token is not at the end of the syllable"
    # Onset, glide, vowel nucleus, then an optional coda: a vowel after a coda consonant means several syllables.
    seen_vowel = seen_coda = False
    for tok in part[:end]:
        base = tok.lstrip("".join(STRESS_MARKS))
        if base in VOWELS:
            if seen_coda:
                return "vowel after a final consonant (glued multi-syllable word?)"
            seen_vowel = True
        elif seen_vowel and base not in GLIDES:
            seen_coda = True
    return None


def validate(phonemes: str, vocab: set[str]) -> Report:
    report = Report()
    tokens = phonemes.split()
    if not tokens:
        report.errors.append("empty phoneme string")
        return report

    oov = sorted({t for t in tokens if t not in vocab})
    if oov:
        report.errors.append(f"tokens not in XPhoneBERT vocab: {oov}")
    not_nfd = sorted({t for t in tokens if unicodedata.normalize("NFD", t) != t})
    if not_nfd:
        report.errors.append(f"tokens not in NFD form: {not_nfd}")

    for i, tok in enumerate(tokens):
        if tok == GLOTTAL and (i == 0 or tokens[i - 1] != NANG):
            report.warnings.append(f"token {i}: 'ʔ' not right after '{NANG}'")
        if tok == "˧":
            report.warnings.append(f"token {i}: lone '˧' (broken tone)")
        if any(mark in tok for mark in STRESS_MARKS):
            report.warnings.append(f"token {i}: stress/length mark in {tok!r}")

    for w, word in enumerate(split_words(phonemes)):
        for s, part in enumerate(split_syllables(word)):
            problem = _check_part(part)
            if problem:
                report.warnings.append(f"word {w} syllable {s} [{' '.join(part)}]: {problem}")
    return report
