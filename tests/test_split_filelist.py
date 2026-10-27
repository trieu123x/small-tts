from pathlib import Path

import pytest

from tts_amis.filelist import read_metadata, write
from tts_amis.split import holdout

METADATA = Path(__file__).resolve().parents[1] / "formatted_dataset2" / "metadata.csv"


def test_holdout_is_deterministic_disjoint_and_sized():
    ids = [wav for wav, _ in read_metadata(METADATA)]
    train, held = holdout(ids)
    assert (len(train), len(held)) == (216, 10)
    assert holdout(ids) == (train, held)
    assert holdout(list(reversed(ids))) == (train, held)
    assert not set(train) & set(held)
    assert held == sorted(held)


def test_holdout_rejects_bad_input():
    with pytest.raises(ValueError):
        holdout(["a", "a", "b"], n=1)
    with pytest.raises(ValueError):
        holdout(["a", "b"], n=2)


def test_read_metadata_whole_dataset():
    rows = read_metadata(METADATA)
    assert len(rows) == 226
    assert rows[0] == ("1.wav", "a-mis kế toán, phần mềm kế toán online.")


def test_read_metadata_malformed_line_names_line(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.wav|ok\n\n2.wav no separator\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"m\.csv:3"):
        read_metadata(p)


def test_write_round_trip(tmp_path):
    p = tmp_path / "train.txt"
    write(p, [("/data/1.wav", "a ˧˧ ▁ m i t ˦˥")])
    assert p.read_text(encoding="utf-8") == "/data/1.wav|a ˧˧ ▁ m i t ˦˥\n"
    with pytest.raises(ValueError):
        write(p, [("/data/1.wav", "a|b")])
