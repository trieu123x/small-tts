"""Alternative Vietnamese G2P tools, called one syllable at a time.

`syllable()` returns the tool's raw output, or None when the tool cannot transcribe the syllable (English words,
names). `to_tokens()` maps raw output onto CharsiuG2P / XPhoneBERT tokens via tts_amis.g2p_map.
"""

import sys
import types
from pathlib import Path

from tts_amis import g2p_map

CHARSIU = "CharsiuG2P"
VPHON = "vPhon"
VIPHONEME = "viphoneme"
ESPEAK = "espeak-ng"
ALL = [CHARSIU, VPHON, VIPHONEME, ESPEAK]


class VPhon:
    """vPhon (kirbyj/vPhon, GPL-3.0), northern dialect, Chao tone numbers.

    Not vendored; pass the folder holding vPhon.py and rules.py.
    """

    def __init__(self, vphon_dir: Path):
        if not (vphon_dir / "vPhon.py").exists():
            raise FileNotFoundError(f"vPhon.py not found in {vphon_dir}; download it from github.com/kirbyj/vPhon")
        sys.path.insert(0, str(vphon_dir))
        import vPhon

        self._vphon = vPhon

    def syllable(self, s: str) -> str | None:
        raw = self._vphon.convert(s, "n", True, False, True, False, False, ".")
        return None if not raw or raw.startswith("[") else raw

    to_tokens = staticmethod(g2p_map.from_vphon)


class Viphoneme:
    """viphoneme 3.0.0 core transcription, northern dialect, Pham tone numbers."""

    def __init__(self):
        # viphoneme imports vinorm at module level only for its own text normalization, which we do ourselves;
        # vinorm itself needs the removed `imp` module and a Linux binary, so stub it out.
        sys.modules["vinorm"] = types.SimpleNamespace(TTSnorm=lambda text: text)
        import viphoneme

        self._vi = viphoneme

    def syllable(self, s: str) -> str | None:
        raw = self._vi.convert(s, "n", 0, 1, 0, 0, ".")
        return None if not raw or raw.startswith("[") else raw

    to_tokens = staticmethod(g2p_map.from_viphoneme)


class Espeak:
    """espeak-ng "vi" through phonemizer, using the library bundled in espeakng-loader (no system install)."""

    def __init__(self):
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer.separator import Separator

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        self._backend = EspeakBackend("vi", language_switch="remove-flags")
        self._sep = Separator(phone=" ", word=" | ")

    def syllable(self, s: str) -> str | None:
        raw = self._backend.phonemize([s], separator=self._sep, strip=True)[0].strip()
        return raw or None

    to_tokens = staticmethod(g2p_map.from_espeak)


def load(name: str, vphon_dir: Path):
    if name == VPHON:
        return VPhon(vphon_dir)
    if name == VIPHONEME:
        return Viphoneme()
    if name == ESPEAK:
        return Espeak()
    raise ValueError(f"unknown G2P backend: {name}")
