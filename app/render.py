"""Render pipeline step outputs as Markdown for the debug UI. Pure Python so it can be tested without torch."""

from html import escape

G2P_LEGEND = (
    "*How to read this:* CharsiuG2P (`text2phonemesequence`) looks each word up in its `vie-n.tsv` dictionary first "
    "(**source = dictionary**) and only runs its ByT5 model for missing words (**source = ByT5 model**). "
    "**ByT5 only** runs the model on every word so you can compare. `▁` separates words, `-` joins syllables. "
    "Tones are Chao tone letters at the end of each syllable: `˧˧` ngang, `˧˨` huyền, `˨˦` sắc, "
    "`˦˥` sắc after p/t/c/ch, `˧˩˨` hỏi, `˧ˀ˥` ngã, `˨ˀ˩` nặng (+ `ʔ` in open syllables).\n\n"
    "*Other G2P tools* (vPhon, viphoneme, espeak-ng) run per syllable; **raw output** is the tool's own notation and "
    "**phonemes → XPhoneBERT** is the same transcription mapped onto Charsiu symbols (tone numbers → Chao letters, "
    "vowel length → a/ă, ɤ/ɤ̆). Symbols with no mapping stay as they are and show up as out-of-vocab in **check**. "
    "Words the tool cannot read fall back to CharsiuG2P. MMS-TTS audio does not depend on the G2P choice."
)


def _cell(value) -> str:
    text = value if isinstance(value, str) else str(value)
    return escape(text).replace("|", "\\|").replace("\n", " ") or " "


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _code(text: str) -> str:
    return f"```\n{text}\n```"


def render_step(name: str, seconds: float, data: dict) -> str:
    title = f"### {name}  ({seconds * 1000:.0f} ms)"
    if name == "normalize":
        body = _table(["sub-step", "output"], [[n, f"`{v}`"] for n, v in data["steps"]])
        if data["dropped"]:
            body += f"\n\n**Dropped characters:** `{' '.join(data['dropped'])}`"
        body += "\n\n**G2P input:**\n" + _code(data["text"])
    elif name == "g2p":
        backend = data.get("backend", "CharsiuG2P")
        body = f"**G2P: {backend}** · full phoneme string sent to XPhoneBERT:\n" + _code(data["phonemes"])
        if not data["aligned"]:
            body += "\n\n⚠️ Number of `▁` chunks differs from the number of input words."
        rows = [
            [
                w["word"],
                w["source"],
                w.get("raw", w["phonemes"]),
                w["phonemes"],
                w.get("tones", ""),
                w.get("compare", ""),
                f"{w['syllables_in']} → {w['syllables_out']}",
                "; ".join(w["problems"]) or "ok",
            ]
            for w in data["words"]
        ]
        compare = data.get("compare_label", "ByT5 only")
        headers = ["word", "source", "raw output", "phonemes → XPhoneBERT", "tones", compare, "syllables", "check"]
        body += "\n\n" + _table(headers, rows) + "\n\n" + G2P_LEGEND
    elif name == "validate":
        lines = [f"- ❌ {e}" for e in data["errors"]] + [f"- ⚠️ {w}" for w in data["warnings"]]
        body = "\n".join(lines) or "✅ No errors or warnings."
    elif name == "xphonebert_tokens":
        body = f"**{data['count']} ids** (including `<s>` / `</s>`)"
        if data.get("expected") is not None and data["count"] != data["expected"]:
            body += f" · ❌ expected {data['expected']} (one per phoneme + 2); the tokenizer dropped or split tokens"
        if data["unk_positions"]:
            body += f" · ❌ unknown tokens at positions {data['unk_positions']}"
        body += "\n\n" + _table(["#", "token", "id"], [[i, t, x] for i, (t, x) in enumerate(data["tokens"])])
    elif name == "xphonebert_encoder":
        body = f"`last_hidden_state` shape **{data['shape']}**, mean {data['mean']}, std {data['std']}"
    elif name == "vits":
        body = (
            f"Checkpoint `{data['checkpoint']}` (epoch {data['epoch']}) · {data['seconds_of_audio']} s of audio "
            f"at {data['sampling_rate']} Hz, peak {data['peak']}"
        )
    elif name == "mms_tts":
        body = (
            f"**MMS input:** `{data['input_text']}`\n\n**MMS tokens:** `{data['tokens']}`\n\n"
            f"{data['seconds_of_audio']} s of audio at {data['sampling_rate']} Hz"
        )
        if data["dropped"]:
            body += f"\n\n⚠️ Characters not in the MMS vocab (silently skipped): `{' '.join(data['dropped'])}`"
    elif name == "ground_truth":
        body = (
            f"Dataset recording `{data['wav']}`: {_cell(data['original'])}"
            if data["wav"]
            else "Input does not match any dataset sentence after normalization, so there is no recording."
        )
    else:
        body = _code(repr(data))
    return f"{title}\n\n{body}"


def render_all(steps: list[tuple[str, float, dict]]) -> str:
    return "\n\n---\n\n".join(render_step(n, s, d) for n, s, d in steps)
