import pytest

from tts_amis.numbers import to_words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "không"),
        (1, "một"),
        (5, "năm"),
        (10, "mười"),
        (11, "mười một"),
        (14, "mười bốn"),
        (15, "mười lăm"),
        (20, "hai mươi"),
        (21, "hai mươi mốt"),
        (24, "hai mươi tư"),
        (25, "hai mươi lăm"),
        (100, "một trăm"),
        (105, "một trăm linh năm"),
        (110, "một trăm mười"),
        (1000, "một nghìn"),
        (1005, "một nghìn không trăm linh năm"),
        (2026, "hai nghìn không trăm hai mươi sáu"),
        (1_000_000, "một triệu"),
        (3_000_021, "ba triệu không trăm hai mươi mốt"),
    ],
)
def test_to_words(n, words):
    assert to_words(n) == words


def test_huge_number_read_digit_by_digit():
    assert to_words(10**12) == "một " + " ".join(["không"] * 12)


def test_negative_rejected():
    with pytest.raises(ValueError):
        to_words(-1)
