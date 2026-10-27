import json
from pathlib import Path

import pytest

from tts_amis.phonemes import split_syllables, split_words, tone_names, validate

PROBE = json.loads(
    (Path(__file__).resolve().parents[1] / "g2p" / "baseline_g2p_probe.json").read_text(encoding="utf-8")
)
# Real XPhoneBERT vocab is only available on Kaggle; every token seen in the probe run is in it (0 OOV there).
VOCAB = {tok for s in PROBE["probes"].values() for tok in s.split()} | {
    tok for row in PROBE["dataset"] for key in ("phon_hyphen_kept", "phon_hyphen_space") for tok in row[key].split()
}

CLEAN_PROBES = [
    "a mít kế toán",
    "a-mít kế toán",
    "tắc",
    "xa",
    "sa la gov",
    "mi-sa-sa-la-gov",
    "hỗ trợ",
    "khuyến mại",
    "hoá đơn",
    "nhiên",
    "quảng",
    "a-mis kế toán , phần mềm kế toán online .",
    "lĩnh vực",
    "mis",
    "gov",
]


@pytest.mark.parametrize("probe", CLEAN_PROBES)
def test_clean_probes_have_no_problems(probe):
    report = validate(PROBE["probes"][probe], VOCAB)
    # "online" in the long probe is a glued two-syllable word, so only check errors there.
    if "online" in probe:
        assert not report.errors
        assert len(report.warnings) == 1
    else:
        assert report.ok, report


@pytest.mark.parametrize("probe", ["marketing", "jetpay", "salagov", "online"])
def test_foreign_words_warn(probe):
    report = validate(PROBE["probes"][probe], VOCAB)
    assert not report.errors
    assert report.warnings


def test_errors():
    assert validate("", VOCAB).errors == ["empty phoneme string"]
    assert "not in XPhoneBERT vocab" in validate("a ˧˧ ▁ zzz", VOCAB).errors[0]
    precomposed = validate("t ă k ˦˥", VOCAB | {"ă"})
    assert any("NFD" in e for e in precomposed.errors)


def test_misplaced_glottal_warns():
    report = validate("c ɤ ˧˧ ʔ", VOCAB)
    assert any("'ʔ' not right after" in w for w in report.warnings)
    assert validate("c ɤ ˨ˀ˩ ʔ", VOCAB).ok


def test_tone_names():
    assert tone_names(PROBE["probes"]["hỗ trợ"]) == ["ngã", "nặng"]
    assert tone_names("t ă k ˦˥ ▁ k e ˨˦") == ["sắc (p/t/c/ch)", "sắc"]
    assert tone_names(PROBE["probes"]["jetpay"]) == ["˧", "ngang"]


def test_split_helpers():
    words = split_words("a ˧˧ - m i t ˦˥ ▁ , ▁ k e ˨˦")
    assert words == [["a", "˧˧", "-", "m", "i", "t", "˦˥"], [","], ["k", "e", "˨˦"]]
    assert split_syllables(words[0]) == [["a", "˧˧"], ["m", "i", "t", "˦˥"]]
    assert split_syllables([","]) == [[","]]
