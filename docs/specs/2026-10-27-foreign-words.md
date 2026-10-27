# Foreign words: respell English and brand names as Vietnamese syllables before G2P

Status: draft · Date: 2026-10-27

## Problem

CharsiuG2P vie-n mangles words that are not Vietnamese syllables. A glued multi-syllable word gets one tone
(`misa → m i s a ˧˧`), English words get stress marks and broken tones (`marketing → m ˧˧ k e t ˈi ŋ ˧˧`), and
acronyms are read as one word. In `formatted_dataset2/metadata.csv`, 48 word types (163 tokens) are written in
English spelling, while annotators respelled 12 others as Vietnamese syllables joined with `-` (`a-mít` ×53,
`xi-a-em`, `hắt-rờ-mờ`). The same name is spelled several ways (`misa` ×32 vs `mi-sa` ×13, `a-mis` ×21 vs
`a-mít` ×53, `we-sign` / `we-sai` / `wi-sai`). User queries will contain more of these (product names, tech words,
acronyms). The current model is `hn_chauanh_v2.pth` (upstream VITS + XPhoneBERT, epoch 330); its training text
front end is unknown.

## Goals

- Every word that is not a Vietnamese syllable is rewritten into Vietnamese syllables joined by `-`, read the way a
  Vietnamese speaker reads it (Vietnamized), before `infer_sentence`.
- Order of sources: curated lexicon → acronym spelling → automatic rules from English pronunciation → spelling-based
  rules for words with no English pronunciation. The source of every rewrite is visible.
- Same rewrite at training and inference time (one function in `tts_amis`).
- The debug app shows the step and can switch it off, so the effect on `hn_chauanh_v2.pth` can be heard A/B.

## Non-goals

- Reading English with English phonemes (eng-us G2P); this voice was trained on Vietnamese only.
- Calling an LLM or any network service at runtime.
- Retraining in this change. A fine-tune from `hn_chauanh_v2.pth` on rewritten transcripts is a follow-up, decided
  after the A/B listening test.
- Number formats beyond what `tts_amis.numbers` does (decimals, dates, currency).

## Design

### 1. Vietnamese syllable check (`tts_amis/syllable.py`)

`is_vietnamese_syllable(s: str) -> bool` on a lowercase NFC syllable, by spelling rules:
`(onset)(w-glide)(nucleus)(coda) + tone`, with
- onsets: `b c ch d đ g gh gi h k kh l m n ng ngh nh p ph qu r s t th tr v x` or none;
- spelling constraints: `k/gh/ngh` only before `i e ê y`; `c/g/ng` not before them; `qu` takes `u` as glide;
- codas: `c ch m n ng nh p t` and glides `i y o u`;
- tone constraint: syllables ending in `p t c ch` take only sắc or nặng.

Used to decide which words are foreign: a word is foreign when any `-`-separated part fails the check.
Known gap: English words that happen to be valid Vietnamese spellings (`in`, `top`, `bum`) pass; the lexicon
overrides them.

### 2. Lexicon (`data/lexicon/foreign.tsv`)

`word<TAB>respelling<TAB>note`, UTF-8, lowercase keys, respelling = valid Vietnamese syllables joined by `-`.
Loaded by `tts_amis/foreign.py`. Seeded with:
- every foreign word type in `metadata.csv` (the 48 found on 2026-10-27), spelled the way the speaker actually says
  it in the recording (checked by listening in the debug app, ground-truth column);
- normalizing the dataset's own variants to one form (`misa → mi-sa`, `a-mis → a-mít`, `we-sign → wi-sai`,
  `net → nét`), choosing the majority spelling;
- MISA product names and common tech words (`email, app, website, zalo, google, facebook, online, admin, ...`).

Lookups are by whole word first, then by `-`-separated part (so `mi-sa-task-go` reuses `task` / `go` entries).

### 3. Acronyms

A token written in ALL CAPS in the raw text (2–6 letters, checked before `lower()`) is spelled letter by letter
from one letter-name table in `tts_amis/foreign.py`, using the dataset's conventions where they exist:
`a ây · b bi · c xi · d đi · e i · f ép · g gi · h hát · i ai · j giây · k cây · l eo · m em · n en · o âu · p pi ·
q kiu · r a · s ét · t ti · u diu · v vi · w đắp-liu · x ích · y quai · z dét`
(`CRM → xi-a-em`, `HRM → hát-a-em`, `KPI → cây-pi-ai`). The lexicon wins over the table (`HRM` may stay
`hắt-rờ-mờ` if that is how MISA says it).

### 4. Automatic rules (fallback, `tts_amis/loanword.py`)

Pure Python, deterministic, every result tagged `auto`.
1. English pronunciation from CMUdict through `eng_to_ipa` (already installed; no network). Map IPA to a sequence of
   consonant/vowel units, drop stress marks.
2. Syllabify with Vietnamese constraints: one onset consonant per syllable (clusters split by inserting `ơ`, e.g.
   `sm → xơ-m`), nucleus from an IPA→Vietnamese vowel table (`ɑ→a, æ→e, ɪ→i, i→i, ɛ→e, ʌ→â, ə→ơ, ɔ→o, u→u, ʊ→u,
   aɪ→ai, eɪ→ây, oʊ→âu, aʊ→ao, ɔɪ→oi`), coda kept only if Vietnamese allows it (`m n ŋ p t k` → `m n ng p t c`),
   otherwise mapped (`s z → t` or dropped, `l → n`/dropped after a vowel, `d → t`, `g → c`, `v f → p`) per a small
   table with tests.
3. Tone: default ngang; stressed closed syllables ending in `p t c` take sắc (`net → nét`, `tax → tắc`), matching
   how the dataset writes them.
4. Words with no CMUdict entry (`gotadi`, `bumas`, `salagov`): spelling-based syllabification of the Latin letters
   with the same tables (`go-ta-đi`, `bu-mát`, `sa-la-gốp`).

The output must pass `is_vietnamese_syllable` for every part; otherwise fall through to letter spelling.

### 5. Pipeline integration

- `tts_amis/text.py`: new step `foreign_words` between `numbers` and `drop_unsupported`, recorded like the other
  steps. `normalize_with_info(raw, foreign=True)` returns, besides the text, a list of
  `Rewrite(word, respelling, source)` where `source ∈ {lexicon, acronym, auto-cmu, auto-spelling, letters}`.
  Acronym detection needs the raw casing, so it runs on the original token before `lower()` and is carried along.
- `app/prepare_filelists.py`: uses the same normalization, so a future fine-tune sees the same text as inference;
  `g2p_report.json` gains `rewrites`.
- Debug app: new `foreign_words` step table (word · respelling · source) and a checkbox "Respell foreign words"
  (default on) so the same sentence can be heard with and without it on `hn_chauanh_v2.pth`.

## Data and API changes

- New file `data/lexicon/foreign.tsv` (in git, public repo: words and respellings only).
- `normalize_with_info` gets a keyword argument `foreign: bool = True` and a `rewrites` field on `Normalized`.
  Existing callers keep working; their output changes for foreign words (intended).
- No model, checkpoint or Kaggle changes.

## Edge cases and errors

- Mixed tokens: `e-shop`, `bank-hắp`, `chấm-net-ích-một`: Vietnamese parts stay, foreign parts are rewritten.
- Already respelled words (`a-mít`, `xi-a-em`) pass the syllable check and are left alone.
- Capitalized sentence-initial Vietnamese words (`Trong`, `Lĩnh`) are not acronyms (acronym = all caps, ≥ 2 letters).
- Single-letter tokens (`a`, `e`, `o`) are Vietnamese words, not acronyms.
- Malformed lexicon line (wrong column count, respelling that fails the syllable check) → error at load time naming
  the line, so a bad entry cannot reach G2P.
- Very long unknown words (> 6 syllables after rules) → letter spelling, and listed in the report.

## Test plan

- `test_syllable.py`: every single-syllable key in `g2p/vie-n.tsv` passes; the 5 known missing syllables
  (`hỗ khuyến nhiên quảng lĩnh`) pass; `misa, marketing, online, salagov, jetpay, sign` fail; `tắc`/`tặc` pass,
  `tảc`/`tàc` fail.
- `test_foreign.py`: lexicon load errors name the line; whole-word then per-part lookup; acronyms (`CRM`, `KPI`,
  `HRM` via lexicon); `Trong`, `A` are not acronyms; dataset variants collapse to one spelling.
- `test_loanword.py`: rule outputs for a fixed list, including the dataset's own respellings as targets
  (`sign → sai`, `net → nét`, `tax → tắc`, `online → on-lai`, `invoice → in-voi`, `amis → a-mít`); every output part
  passes `is_vietnamese_syllable`.
- `test_text.py`: normalizing all 226 dataset lines with `foreign=True` leaves no word that fails the syllable check;
  `foreign=False` keeps today's output exactly.
- End to end: in the debug app with `--checkpoint hn_chauanh_v2.pth`, the 10 holdout sentences and 10 new sentences
  with English words synthesize with the checkbox on and off; G2P validation shows 0 "glued multi-syllable" warnings
  with it on.

## Acceptance criteria

- [ ] `.venv/Scripts/python -m pytest` and `ruff check .` pass.
- [ ] `normalize_with_info(text, foreign=True)` on all 226 `metadata.csv` lines yields only words whose every part
      passes `is_vietnamese_syllable` (test).
- [ ] Running `app.prepare_filelists` with the rewrite gives 0 rows with "glued multi-syllable" warnings (was 71).
- [ ] `data/lexicon/foreign.tsv` covers all 48 foreign word types found in the dataset.
- [ ] The debug app shows the `foreign_words` step with a source per rewrite, and the checkbox switches it off.
- [ ] `normalize_with_info(text, foreign=False)` output equals today's output for all 226 lines (test).

## Open questions

- How `hn_chauanh_v2.pth` was trained is unknown (text front end, data). The A/B checkbox answers the practical
  question (does respelled input sound better with this checkpoint?); a fine-tune is decided after that.
- Letter names: the table above follows the dataset where it has examples; confirm the rest (`h` hát vs ếch,
  `w` đắp-liu vs vê-kép, `r` a vs rờ).
- Which respelling is canonical when the recordings disagree with the transcripts needs listening; the user
  confirms the seed lexicon before it is used for any fine-tune.
