import json
import unicodedata
from pathlib import Path

import pytest

from tts_amis.g2p_map import from_espeak, from_viphoneme, from_vphon

PROBE_FILE = Path(__file__).resolve().parents[1] / "g2p" / "baseline_g2p_probe.json"
PROBE = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
# Every token CharsiuG2P produced in the probe run; all of them are in the XPhoneBERT vocab.
CHARSIU_TOKENS = {tok for row in PROBE["dataset"] for tok in row["phon_hyphen_kept"].split()}


def check(tokens: list[str], expected: str) -> None:
    assert tokens == unicodedata.normalize("NFD", expected).split()
    assert set(tokens) <= CHARSIU_TOKENS, set(tokens) - CHARSIU_TOKENS


# Raw outputs recorded from the real tools:
# vPhon 076a5d3 (-d n -c -n), viphoneme 3.0.0, espeak-ng via espeakng-loader 0.2.4.


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        (".m.iː.33.", "m i ˧˧"),
        (".t.w.aː.n.24.", "t w a n ˨˦"),
        (".t.a.k.45.", "t ă k ˦˥"),
        (".tɕ.əː.21g.", "c ɤ ˨ˀ˩ ʔ"),
        (".m.oː.t.21.", "m o t ˨ˀ˩"),
        (".h.oː.3g5.", "h o ˧ˀ˥"),
        (".k.w.aː.ŋ.312.", "k w a ŋ ˧˩˨"),
        (".h.aː.ŋ.32.", "h a ŋ ˧˨"),
        (".ŋ.iə.ŋ.33.", "ŋ i ə ŋ ˧˧"),
        (".v.ɨ.k.21.", "v ɯ k ˨ˀ˩"),
        (".th.w.iː.312.", "tʰ w i ˧˩˨"),
    ],
)
def test_vphon(raw, tokens):
    check(from_vphon(raw), tokens)


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        (".m.i.1.", "m i ˧˧"),
        (".tw.ʷa.n.5.", "t w a n ˨˦"),
        (".t.ă.k.5.", "t ă k ˦˥"),
        (".ʈ.ɤ.6.", "c ɤ ˨ˀ˩ ʔ"),
        (".m.o.t.6.", "m o t ˨ˀ˩"),
        (".h.o.3.", "h o ˧ˀ˥"),
        (".kw.a.ŋ.4.", "k w a ŋ ˧˩˨"),
        (".tʰw.ʷi.4.", "tʰ w i ˧˩˨"),
        (".x.a.tʃ.5.", "x a c ˦˥"),
        (".ʂ.a.1.", "s a ˧˧"),
        (".j.a.5.", "z a ˨˦"),
        (".ɤ̆.j.5.", "ɤ̆ j ˨˦"),
        (".xw.ʷiə.n.5.", "x w i ə n ˨˦"),
    ],
)
def test_viphoneme(raw, tokens):
    check(from_viphoneme(raw), tokens)


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        ("m i", "m i ˧˧"),
        ("t̪ w aːɜ n", "t w a n ˨˦"),
        ("t̪ aɜ c", "t ă c ˦˥"),
        ("tʃ əː6", "c ɤ ˨ˀ˩ ʔ"),
        ("m o6 t̪", "m o t ˨ˀ˩"),
        ("h o5", "h o ˧ˀ˥"),
        ("k w aː4 ŋ", "k w a ŋ ˧˩˨"),
        ("h aː2 ŋ", "h a ŋ ˧˨"),
        ("ŋ iɛ ŋ", "ŋ i ə ŋ ˧˧"),
        ("v y6 c", "v ɯ c ˨ˀ˩"),
    ],
)
def test_espeak(raw, tokens):
    check(from_espeak(raw), tokens)


def test_unknown_symbols_pass_through():
    assert from_espeak("m ɪ s eɪ") == ["m", "ɪ", "s", "eɪ", "˧˧"]
    assert from_vphon(".x.99.") == ["x", "99"]
    assert from_vphon("") == [] and from_viphoneme("") == [] and from_espeak("") == []
