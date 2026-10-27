import re
import unicodedata
from pathlib import Path

import pytest

from tts_amis.text import normalize, normalize_with_info

METADATA = Path(__file__).resolve().parents[1] / "formatted_dataset2" / "metadata.csv"


def test_punctuation_split_and_lower():
    assert normalize("A-mis kế toán, phần mềm online.") == "a-mis kế toán , phần mềm online ."


def test_nfd_input_gives_nfc_result():
    raw = "hóa đơn điện tử"
    assert normalize(unicodedata.normalize("NFD", raw)) == normalize(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("hóa đơn", "hoá đơn"), ("thủy", "thuỷ"), ("khỏe", "khoẻ"), ("Hòa", "hoà"), ("hoàng", "hoàng"), ("của", "của")],
)
def test_tone_placement(raw, expected):
    assert normalize(raw) == expected


def test_digits_glued_to_hyphen_stay_one_word():
    assert normalize("chấm-net-ích-1, nâng cấp") == "chấm-net-ích-một , nâng cấp"
    assert normalize("bản 21") == "bản hai-mươi-mốt" or normalize("bản 21") == "bản hai mươi mốt"


def test_digits_glued_to_letters_get_spaces():
    assert normalize("abc12") == "abc mười hai"


def test_unsupported_chars_dropped_and_reported():
    result = normalize_with_info("giảm 50% (nhanh) & rẻ")
    assert result.text == "giảm năm mươi nhanh rẻ"
    assert result.dropped == ["%", "&", "(", ")"]


def test_steps_are_recorded_in_order():
    names = [name for name, _ in normalize_with_info("Một.").steps]
    assert names == ["nfc_whitespace", "lower", "tone_placement", "numbers", "drop_unsupported", "split_punctuation"]


@pytest.mark.parametrize("raw", ["", "   ", "...", "!?"])
def test_no_letters_rejected(raw):
    with pytest.raises(ValueError):
        normalize(raw)


def test_whole_dataset_normalizes_cleanly():
    lines = [line for line in METADATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 226
    for line in lines:
        out = normalize(line.split("|", 1)[1])
        assert not re.search(r"\d", out), out
        assert not re.search(r"\w[,.!?;:]|[,.!?;:]\w", out), out
        assert out == out.lower()
