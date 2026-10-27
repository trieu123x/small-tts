"""Build VITS filelists from metadata.csv: normalize -> CharsiuG2P -> validate -> split -> write.

Same G2P calls as upstream preprocess.py (dictionary lookup, ByT5 fallback, `segments` split), but per sentence with
our normalization in front. Exits non-zero on any error so a training kernel never starts on bad data.

    python -m app.prepare_filelists --metadata data/metadata.csv --wav-dir data/wavs --out-dir filelists
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from text2phonemesequence import Text2PhonemeSequence

from app.xphonebert_tokenizer import load_xphonebert_tokenizer
from tts_amis.filelist import read_metadata, write
from tts_amis.phonemes import validate
from tts_amis.split import holdout
from tts_amis.text import normalize_with_info

MAX_XPHONEBERT_TOKENS = 512  # position embeddings of vinai/xphonebert-base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument("--holdout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke tests)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    rows = read_metadata(args.metadata)
    if args.limit:
        rows = rows[: args.limit]
    vocab = set(load_xphonebert_tokenizer().get_vocab())
    g2p = Text2PhonemeSequence(language="vie-n", is_cuda=args.cuda)
    if len(g2p.phone_dict) < 70000:
        print(f"vie-n.tsv not loaded (dict size {len(g2p.phone_dict)}); put it in the CWD", file=sys.stderr)
        return 1

    report, failed = {}, []
    for wav, text in rows:
        entry: dict = {"text": text, "errors": [], "warnings": []}
        report[wav] = entry
        path = args.wav_dir / wav
        if not path.exists():
            entry["errors"].append(f"missing wav {path}")
        else:
            info = sf.info(path)
            if info.samplerate != args.sampling_rate or info.channels != 1:
                entry["errors"].append(f"wav is {info.samplerate} Hz x{info.channels}, need {args.sampling_rate} mono")
            entry["seconds"] = round(info.duration, 2)
        try:
            norm = normalize_with_info(text)
        except ValueError as e:
            entry["errors"].append(str(e))
            failed.append(wav)
            continue
        phonemes = g2p.infer_sentence(norm.text)
        check = validate(phonemes, vocab)
        n_tokens = len(phonemes.split()) + 2
        if n_tokens > MAX_XPHONEBERT_TOKENS:
            check.errors.append(f"{n_tokens} tokens exceed XPhoneBERT's {MAX_XPHONEBERT_TOKENS}")
        entry.update(normalized=norm.text, dropped=norm.dropped, phonemes=phonemes, tokens=n_tokens)
        entry["errors"] += check.errors
        entry["warnings"] += check.warnings
        if entry["errors"]:
            failed.append(wav)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "g2p_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    n_warn = sum(1 for e in report.values() if e["warnings"])
    print(f"{len(rows)} rows, {len(failed)} with errors, {n_warn} with warnings")
    if failed:
        for wav in failed:
            print(f"ERROR {wav}: {report[wav]['errors']}", file=sys.stderr)
        return 1

    train_ids, val_ids = holdout([wav for wav, _ in rows], n=args.holdout)
    wav_dir = args.wav_dir.resolve()
    for name, ids in (("amis_train.txt", train_ids), ("amis_val.txt", val_ids)):
        write(args.out_dir / name, [((wav_dir / i).as_posix(), report[i]["phonemes"]) for i in ids])
    print(f"wrote {len(train_ids)} train / {len(val_ids)} val lines to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
