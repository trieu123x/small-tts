# Baseline TTS pipeline (CharsiuG2P + XPhoneBERT + VITS)

Status: draft · Date: 2026-10-27

## Problem

We want to know how good a plain VITS + XPhoneBERT model trained on our 226-utterance MISA dataset (13.6 min, 24 kHz mono) is,
before spending time on better text processing or more data. No code exists yet. Upstream has no Vietnamese TTS checkpoint,
and torch cannot run on the local Windows machine (App Control), so training and inference run as Kaggle kernels.

## Goals

- A reproducible pipeline: raw `metadata.csv` text → normalized G2P input → CharsiuG2P phonemes → XPhoneBERT tokens → VITS training.
- One training run of ~10 h on one Kaggle GPU session, producing a checkpoint as kernel output.
- A batch inference kernel that synthesizes the 10 held-out sentences and produces `index.html` with generated audio,
  ground-truth audio, normalized text, phonemes and validation warnings side by side. The user downloads it and listens locally.
- A no-training reference: the same page has a column from the pretrained `facebook/mms-tts-vie` (VITS, character input,
  generic voice, CC-BY-NC). The inference kernel can run before training finishes and then shows only MMS + ground truth.
- Pure-Python parts (normalization, phoneme validation, split, filelists, report) are unit-tested locally without torch.

## Non-goals

- Quality improvements: proper-name rewriting (`salagov → sa-la-gov`), VnCoreNLP word segmentation, pretraining on public data.
- Interactive UI (Gradio), arbitrary user text at runtime, local inference, ONNX export.
- Fine-tuning MMS-TTS on our data, or using it for anything beyond internal comparison.
- Multi-speaker, objective metrics (MOS, CER via ASR).
- Running the vendored upstream code locally.

## Design

### Repository and code delivery

- Kernels do not clone GitHub: `kaggle/build_kernel.py` embeds the files listed in each `kaggle/<kernel>/kernel.json`
  (`include`) into a single script kernel, so a run never depends on what has been pushed. (Changed 2026-10-27 from the
  original "clone small-tts" plan.) The project still goes to the public repo `https://github.com/trieu123x/small-tts`;
  never commit wavs, checkpoints, zip or the Kaggle token (covered by `.gitignore`).
- Data goes to a private Kaggle dataset `triuinhhi/tts-amis-data` containing `metadata.csv` and `wavs/` (from `formatted_dataset2/`).

### Layout (new files)

```
pyproject.toml                    # ruff + pytest config only (no packaging)
requirements-dev.txt              # pytest, ruff
tts_amis/
  __init__.py
  numbers.py                      # int → Vietnamese words
  text.py                         # normalize(text) → G2P input string
  phonemes.py                     # validate(phonemes, vocab) → Report
  split.py                        # deterministic holdout split
  filelist.py                     # read metadata, write VITS filelists
  report.py                       # build index.html from results
tests/                            # pytest, one file per module
third_party/vits_xphonebert/      # vendored VinAIResearch/XPhoneBERT@7bd41b9 VITS_with_XPhoneBERT, patched
  UPSTREAM.md                     # source URL, commit, list of local patches
  configs/amis_base_xphonebert.json
kaggle/
  train/kernel-metadata.json, train.py
  infer/kernel-metadata.json, infer.py
```

`tts_amis` must stay importable on Kaggle's Python (3.10+ syntax only) and must not import torch or transformers.

### Text normalization (`tts_amis/text.py`)

`normalize(text: str) -> str`, applied in this order:
1. `unicodedata.normalize("NFC")`, collapse whitespace, strip.
2. Tone-mark placement to the **new style** (`hóa → hoá`, `thủy → thuỷ`, `khỏe → khoẻ`), which matches more `vie-n.tsv` keys.
3. `lower()`.
4. Digit runs → Vietnamese words via `numbers.to_words(int)` (`1 → một`, `21 → hai mươi mốt`, `105 → một trăm linh năm`).
   A digit run glued to `-` (`ích-1`) becomes its own syllable (`ích-một`).
5. Split punctuation `, . ! ? ; :` into separate tokens: `re.sub(r"\s*([,.!?;:])\s*", r" \1 ", t)`, then collapse spaces.
6. Keep `-` between syllables as in the dataset. No `_`, no word segmentation. English words and names are left as they are.
7. Any other character not in `[a-zđ + Vietnamese vowels with marks] - , . ! ? ; : space` is dropped and reported (returned in a
   `dropped` list from `normalize_with_info(text) -> (str, list[str])`).

Empty or whitespace-only input → `ValueError`.

### Phoneme validation (`tts_amis/phonemes.py`)

`validate(phonemes: str, vocab: set[str]) -> Report` where `Report` holds `errors: list[str]` and `warnings: list[str]`.
- **Errors** (block training): empty string; any token not in `vocab`; precomposed `ă` (U+0103) or other non-NFD form.
- **Warnings** (reported, do not block): a `▁`-chunk or `-`-part that is not punctuation and does not contain exactly one tone token
  from `{˧˧ ˧˨ ˨˦ ˦˥ ˧˩˨ ˧ˀ˥ ˨ˀ˩}` at its end (optionally followed by `ʔ`); `ʔ` not right after `˨ˀ˩`; any of `ˈ ˌ ː`;
  a lone `˧`.

Locally, `vocab` in tests comes from the token set in `g2p/baseline_g2p_probe.json`. On Kaggle it is the real XPhoneBERT vocab
(`AutoTokenizer.from_pretrained("vinai/xphonebert-base").get_vocab()`).

### Split and filelists

- `split.holdout(ids: list[str], n=10, seed=1234) -> (train_ids, holdout_ids)`: deterministic, disjoint, sorted output.
  The holdout set doubles as the VITS validation set (used only for logging, never for gradients).
- `filelist.read_metadata(path) -> list[tuple[str, str]]` parses `N.wav|text` (UTF-8, skips blank lines, errors on malformed lines
  with the line number).
- `filelist.write(path, rows: list[tuple[str, str]])` writes `<abs wav path>|<phonemes>` lines.

### Vendored VITS (`third_party/vits_xphonebert/`)

Copy `VITS_with_XPhoneBERT` at commit `7bd41b9912cc356a89e7de7ac596f42ba8b932c4` with its MIT `LICENSE`. Exclude `__pycache__`,
`build/`, prebuilt `.so`, the LJSpeech filelists and `requirements.txt`. Local patches, each listed in `UPSTREAM.md`:
- Compatibility with Kaggle's current torch (e.g. `torch.stft(..., return_complex=True)` + `torch.view_as_real`, deprecated APIs).
- `train.py`: `--max_minutes` wall-clock limit; on hitting it, save a final checkpoint and exit cleanly.
  Keep only the latest `G_*.pth`/`D_*.pth` pair on disk (each pair is ~1 GB with optimizer state; Kaggle output limit is 20 GB).
- DDP on 2 GPUs (Kaggle T4×2), still working on 1 GPU (`torch.cuda.device_count()`). `batch_size` in the config is per GPU.
- Fix upstream `inference.py` (`get_text` vs `get_inputs` mismatch) or replace it with `kaggle/infer/infer.py`.

`configs/amis_base_xphonebert.json` is based on `lj_base_xphonebert.json` with `sampling_rate: 24000` (hop 256, upsample product 256
unchanged), batch size reduced to fit a 16 GB GPU (start at 16, tune in the smoke run), and `bert: vinai/xphonebert-base`.

### Kaggle training kernel (`kaggle/train/train.py`)

Kernel `triuinhhi/tts-amis-train`, GPU T4×2, internet on, dataset source `triuinhhi/tts-amis-data`. Steps:
1. `git clone` small-tts, log the SHA to `git_sha.txt`.
2. `pip install text2phonemesequence==0.1.4 cython`, build `monotonic_align` in place. Use Kaggle's preinstalled torch/transformers.
3. Copy `g2p/vie-n.tsv` into the CWD (the library's `wget` is unreliable).
4. For every metadata row: `normalize` → `Text2PhonemeSequence(language="vie-n").infer_sentence` → `validate`.
   Write `g2p_report.json` (per id: text, normalized, phonemes, errors, warnings, dropped chars). Abort if any row has errors.
5. Split, write `train.txt` and `val.txt` filelists. Assert holdout ids are absent from `train.txt`.
6. Run vendored `train.py` with `--max_minutes 600`.
7. Copy to `/kaggle/working/`: `G_latest.pth`, `config.json`, filelists, `g2p_report.json`, `git_sha.txt`, `train_log.txt`.

`MAX_MINUTES` is a constant at the top of the script so a short smoke run (e.g. 10) uses the same code path.

### Kaggle inference kernel (`kaggle/infer/infer.py`)

Kernel `triuinhhi/tts-amis-infer`, GPU, dataset source `triuinhhi/tts-amis-data`, and kernel source `triuinhhi/tts-amis-train`
(for the checkpoint) once a training run exists. Two modes, picked automatically:
- **MMS only**: no `G_latest.pth` found under `/kaggle/input` (training not run yet). The holdout ids come from `tts_amis.split.holdout`
  over `metadata.csv`, which gives the same 10 ids as the training kernel.
- **Full**: checkpoint found; the holdout ids come from its `val.txt`.

Steps:
1. Clone small-tts, install and set up as in training steps 1–3.
2. For each of the 10 holdout rows:
   - VITS + XPhoneBERT (full mode only): normalize → G2P → validate → XPhoneBERT tokenize →
     `net_g.infer(noise_scale=.667, noise_scale_w=.8, length_scale=1)` → `out/vits/<id>.wav` (24 kHz).
   - MMS: `tts_amis.text.normalize` → `-` replaced by a space → characters missing from the MMS tokenizer vocab dropped and recorded →
     `VitsModel.from_pretrained("facebook/mms-tts-vie")` with a fixed seed → `out/mms/<id>.wav` (16 kHz, the model's native rate).
   - Copy the ground-truth wav to `out/gt/<id>.wav`.
3. Write `results.json` and `out/index.html` via `tts_amis.report.build(results, path)`. One row per id with columns:
   original text, normalized text, phonemes + warnings, GT audio, VITS audio (or "not trained yet"), MMS audio (+ dropped chars).

The user downloads with `kaggle kernels output triuinhhi/tts-amis-infer -p runs/<date>/` (add `runs/` to `.gitignore`) and opens
`index.html` in a browser.

### Docs

- Update `.claude/CLAUDE.md`: fill verified commands, record the decision "baseline uses no word segmentation", add the Kaggle run commands.

## Data and API changes

- New private Kaggle dataset `triuinhhi/tts-amis-data`, new kernels `triuinhhi/tts-amis-train` and `triuinhhi/tts-amis-infer`.
- New git remote `origin` → `trieu123x/small-tts` (public).
- `.gitignore`: add `runs/`.
- No other schema or API changes.

## Edge cases and errors

- Empty or punctuation-only text → `ValueError` locally; on Kaggle the row is recorded as an error and training aborts.
- Unsupported characters (`%`, `&`, `/`, emoji) → dropped and listed in the report; not fatal.
- Glued English/multi-syllable words → validation warnings, not fatal (baseline keeps them on purpose).
- Missing wav for a metadata row, or wav not 24 kHz mono → training kernel aborts before training with the offending ids.
- `vie-n.tsv` missing in CWD → kernel aborts (don't let G2P silently fall back to the model for every word).
- Training killed by the 12 h Kaggle limit → avoided by `--max_minutes 600`; the latest periodic checkpoint is still in the output.
- Out of GPU memory → lower `batch_size` in the config (found in the smoke run).
- Inference on a sentence longer than the model can handle → synthesize anyway and record duration; no truncation.
- Generated audio silent or NaN → recorded as a failure in `results.json` and shown in `index.html`, not fatal for other rows.

## Test plan

Local unit tests (`.venv/Scripts/python -m pytest`):
- `test_numbers.py`: 0, 1, 5, 10, 11, 15, 21, 24, 105, 1000, 2026, 1 000 000; negative input → `ValueError`.
- `test_text.py`: punctuation split (`toán,` → `toán ,`); NFD input equals NFC result; `hóa`/`thủy` → `hoá`/`thuỷ`;
  uppercase lowered; `-` kept; `ích-1` → `ích-một`; dropped chars reported; empty → `ValueError`;
  normalizing all 226 `metadata.csv` lines leaves no glued punctuation and no digits.
- `test_phonemes.py`: every probe in `baseline_g2p_probe.json` with no English words validates with no errors and no warnings;
  `marketing`, `jetpay`, `salagov` outputs produce warnings; unknown token, U+0103 `ă`, empty → errors; misplaced `ʔ` → warning.
- `test_split.py`: deterministic across calls, disjoint, sizes 216/10, holdout sorted.
- `test_filelist.py`: round-trip of `metadata.csv` (226 rows); malformed line error names the line number.
- `test_report.py`: HTML contains one row per result, escapes `<`/`&` in text, references `vits/<id>.wav`, `mms/<id>.wav`
  and `gt/<id>.wav`; a result without VITS audio shows "not trained yet" instead of a broken player.

End-to-end (Kaggle):
1. MMS only: push the inference kernel before any training. Status `complete`; `index.html` has 10 rows with GT and MMS audio.
2. Smoke: push the training kernel with `MAX_MINUTES = 10`, then the inference kernel. Both finish with status `complete`,
   and `index.html` has 10 rows with GT, VITS (noise is fine at this point) and MMS audio.
3. Full: same with `MAX_MINUTES = 600`; listen to `index.html`.

## Acceptance criteria

- [ ] `.venv/Scripts/python -m pytest` passes and `.venv/Scripts/python -m ruff check .` reports no issues.
- [ ] `tts_amis` has no import of torch, transformers or text2phonemesequence (checked by grep in a test).
- [ ] `third_party/vits_xphonebert/UPSTREAM.md` names commit `7bd41b9` and every patched file; `LICENSE` is present.
- [ ] Training kernel status is `complete`; its output contains `G_latest.pth`, `config.json`, `train.txt`, `val.txt`,
      `g2p_report.json`, `git_sha.txt`, `train_log.txt`.
- [ ] `g2p_report.json` has 226 entries and 0 errors; `train.txt` has 216 lines and shares no id with `val.txt` (10 lines).
- [ ] Inference kernel in MMS-only mode (no training output attached) has status `complete`; output has 10 MMS wavs at 16000 Hz
      and 10 ground-truth wavs, `results.json` and `index.html`.
- [ ] Inference kernel in full mode has status `complete`; output has 10 VITS wavs at 24000 Hz and 10 MMS wavs at 16000 Hz,
      each between 0.3 s and 30 s and not all-zero, plus 10 ground-truth wavs, `results.json` and `index.html`.
- [ ] The 10 holdout ids are identical in MMS-only mode, full mode and the training kernel's `val.txt`.
- [ ] Full run trained for at least 9 h of wall clock (from `train_log.txt`) without hitting the 12 h kill.

## Open questions

None. Resolved: `metadata.csv` and `g2p/vie-n.tsv` may be public; train on T4×2; no extra sentence list in this round;
MMS-TTS is added as a no-training reference (CC-BY-NC: internal comparison only, not for shipping).
