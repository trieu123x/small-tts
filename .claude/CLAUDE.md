# tts-amis

Vietnamese TTS for MISA product names and phrases: fine-tune VITS with XPhoneBERT as the text encoder on a small single-speaker dataset.

Early stage: no code yet. The plan and phoneme format live in `g2p/PHONEME_FORMAT.md` (Vietnamese). Read it before any text/G2P work.

## Stack

- Python. Local code: text normalization, G2P preprocessing, filelist validation (pure Python, no torch).
- Training/inference: VITS + XPhoneBERT (`vinai/xphonebert-base`), G2P via `text2phonemesequence==0.1.4` (`charsiu/g2p_multilingual_byT5_small_100`). Runs on Kaggle only.
- Tooling: pip + venv, pytest, ruff (lint + format).

## Commands

All run from the repo root (Git Bash or PowerShell).
- Install: `python -m venv .venv` then `.venv/Scripts/python -m pip install -r requirements-dev.txt`
- Test (all): `.venv/Scripts/python -m pytest`
- Test (one): `.venv/Scripts/python -m pytest tests/test_text.py::test_tone_placement`
- Lint: `.venv/Scripts/python -m ruff check .`
- Format: `.venv/Scripts/python -m ruff format .`
- Kaggle kernel (torch code): `.venv/Scripts/python kaggle/build_kernel.py train` then
  `python -m kaggle kernels push -p kaggle/train/build`; follow with `python -m kaggle kernels status|logs|output <id>`.
  Needs env var `KAGGLE_API_TOKEN` (Windows user env). Training length: `MAX_MINUTES` / `EPOCHS` in `kaggle/train/main.py`.
- VITS CPU smoke (2 train steps + inference, ~20 s): `.venv/Scripts/python -m app.prepare_filelists ...` (run from
  `runs/local-app/`, see its docstring) then `.venv/Scripts/python scripts/vits_cpu_smoke.py --filelist
  runs/vits-smoke/filelists/amis_train.txt --work runs/vits-smoke` (env `HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8`).

## Layout

- `tts_amis/`: pure Python (normalization, number reading, phoneme validation). Must not import torch.
- `app/`: torch code (`pipeline.py`) and the Gradio debug app; `app/render.py` stays pure Python for local tests.
- `third_party/vits_xphonebert/`: vendored upstream VITS+XPhoneBERT, patched; every change is listed in its
  `UPSTREAM.md` and marked `tts-amis patch`. Patch with minimal edits, never reformat.
- `kaggle/<kernel>/kernel.json` + `kaggle/build_kernel.py`: bundle the files listed in `include` into one script kernel
  (output in `kaggle/<kernel>/build/`, generated, gitignored). Data comes from private dataset `triuinhhi/tts-amis-data`.

- `formatted_dataset2/metadata.csv`: `N.wav|text`, 226 lines. `wavs/`: 24 kHz mono 16-bit PCM. Source data, don't edit.
- `g2p/vie-n.tsv`: CharsiuG2P vie-n dictionary (`word<TAB>ipa`, NFC, 70,901 lines). Reference copy, don't edit.
- `g2p/baseline_g2p_probe.json`: baseline G2P output for probes and all 226 sentences. Use as a regression fixture.

## Conventions

- Target filelist line: `N.wav|<phonemes>`, tokens separated by single spaces, word boundary token `▁` (U+2581).
- Phoneme strings are NFD (`ă` = `a` + U+0306). Precomposed U+0103 is not in the XPhoneBERT vocab.
- Text preprocessing order before `infer_sentence`: NFC → normalize tone-mark placement → `lower()` → numbers to words → split punctuation into separate tokens → rewrite names/English as valid VN syllables joined with `-`.
- Filelist invariants (enforce in tests): every token in the XPhoneBERT vocab; exactly one tone token per syllable from `˧˧ ˧˨ ˨˦ ˦˥ ˧˩˨ ˧ˀ˥ ˨ˀ˩`; `ʔ` only right after `˨ˀ˩`; no `ˈ ˌ ː`; no lone `˧`.
- Validate syllables by spelling rules (onset + rhyme + tone, sắc/nặng with stops p/t/c/ch), not by `vie-n.tsv` as a whitelist: the dictionary is incomplete and mixes old/new tone-mark placement (`hoá` vs `hóa`).

## Gotchas

- `.venv` is Python 3.13 with CPU torch, transformers, gradio and text2phonemesequence. Don't use Python 3.15:
  transformers model loading segfaults there. Training still goes to Kaggle (no local GPU); unit tests must not import torch.
- Never load the XPhoneBERT tokenizer with `AutoTokenizer` (transformers 5.x makes it a byte-level BPE RobertaTokenizer
  that silently drops tones, `▁`, `ʔ`, `ɤ̆`). Use `app.pipeline.load_xphonebert_tokenizer()` (WordLevel from tokenizer.json).
- Local debug app: from `runs/local-app/` (holds a copy of `vie-n.tsv`), run
  `HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONPATH=<repo> ../../.venv/Scripts/python -m app.debug_app --data-dir ../../formatted_dataset2`
  → http://127.0.0.1:7860. G2P radio: CharsiuG2P / vPhon / viphoneme / espeak-ng. vPhon (GPL-3.0, kirbyj/vPhon@076a5d3)
  lives in `runs/local-app/vphon/`, not in the repo; viphoneme runs with a stubbed `vinorm` (broken on Py ≥ 3.12).
- `text2phonemesequence` downloads `vie-n.tsv` with `wget` into the CWD and fails silently without it (empty dict, everything goes through ByT5). Copy `g2p/vie-n.tsv` into the CWD first.
- `infer_sentence` does not lowercase, and punctuation glued to a word (`toán,`) is silently dropped. Digits come out wrong (`1 → ɛ ˧˧`).
- Multi-syllable words written together (`salagov`, `misa`) get one tone for the whole word. Always hyphenate (`sa-la-gov`).
- Word segmentation (none vs VnCoreNLP `_`) is still undecided; keep train and inference consistent.
- Local Python is 3.15; Kaggle's is older. Don't use syntax newer than Kaggle's Python in code that runs there.

## How we work

- Talk to me in Vietnamese. Write code, comments, commit messages and docs in English.
- Big or unclear work: `/spec` first, then `/feature <spec>`. Bugs: `/fix`. Before committing: `/ship`.
- If the diff fits in one sentence, skip the plan and just do it.
- Prove that work is done with evidence: the command you ran and its result. Never say "should work".
- Use the `test-runner` subagent for test or build runs with long output, `reviewer` before committing, `security-reviewer` when touching auth, input parsing, SQL, shell or network code, and `docs-researcher` for library questions.
- Follow the patterns already in the codebase. Don't add a dependency, change public APIs or touch unrelated code without asking.
- When compacting, keep the list of modified files, the active spec path, the test commands and any failing tests.
