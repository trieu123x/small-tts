from app.render import render_all, render_step


def test_g2p_table_escapes_and_lists_words():
    data = {
        "phonemes": "a ˧˧ ▁ , ▁ x <y>",
        "aligned": True,
        "words": [
            {
                "word": "a",
                "source": "dictionary",
                "phonemes": "a ˧˧",
                "syllables_in": 1,
                "syllables_out": 1,
                "problems": [],
            },
            {
                "word": "a|b",
                "source": "ByT5 model",
                "phonemes": "x <y>",
                "syllables_in": 1,
                "syllables_out": 1,
                "problems": ["no tone token"],
            },
        ],
    }
    out = render_step("g2p", 0.012, data)
    assert "(12 ms)" in out
    assert "a\\|b" in out
    assert "x &lt;y&gt;" in out
    assert "no tone token" in out


def test_validate_ok_and_problems():
    assert "No errors" in render_step("validate", 0, {"errors": [], "warnings": []})
    out = render_step("validate", 0, {"errors": ["bad"], "warnings": ["meh"]})
    assert "❌ bad" in out and "⚠️ meh" in out


def test_token_count_mismatch_is_flagged():
    data = {"count": 4, "expected": 6, "unk_positions": [], "tokens": [("<s>", 0), ("k", 13), ("e", 14), ("</s>", 2)]}
    assert "expected 6" in render_step("xphonebert_tokens", 0, data)
    assert "expected" not in render_step("xphonebert_tokens", 0, {**data, "expected": 4})


def test_ground_truth_missing():
    assert "no recording" in render_step("ground_truth", 0, {"wav": None, "original": None})


def test_render_all_joins_steps():
    out = render_all([("validate", 0, {"errors": [], "warnings": []}), ("unknown", 0, {"k": 1})])
    assert out.count("###") == 2


def test_vits_step():
    data = {"checkpoint": "G_1.pth", "epoch": 330, "sampling_rate": 24000, "seconds_of_audio": 1.95, "peak": 0.2}
    out = render_step("vits", 0, data)
    assert "`G_1.pth` (epoch 330)" in out and "24000 Hz" in out
