"""Run every TTS pipeline step on one sentence and keep each step's output. Needs torch; runs on Kaggle."""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from text2phonemesequence import Text2PhonemeSequence
from transformers import AutoModel, AutoTokenizer, VitsModel

from app import g2p_backends
from app.xphonebert_tokenizer import XPHONEBERT, load_xphonebert_tokenizer
from tts_amis.phonemes import PUNCTUATION_TOKENS, split_syllables, split_words, tone_names, validate
from tts_amis.text import normalize, normalize_with_info

MMS = "facebook/mms-tts-vie"
VITS_DIR = Path(__file__).resolve().parents[1] / "third_party" / "vits_xphonebert"
VITS_CONFIG = VITS_DIR / "configs" / "amis_base_xphonebert.json"


def load_vits(checkpoint: Path, config: Path = VITS_CONFIG):
    """Build the vendored SynthesizerTrn from `config` and load a trained G_*.pth into it (strict: no missing keys)."""
    if str(VITS_DIR) not in sys.path:
        sys.path.insert(0, str(VITS_DIR))  # upstream modules import each other by bare name
    import utils
    from models import SynthesizerTrn

    hps = utils.get_hparams_from_file(str(config))
    net_g = SynthesizerTrn(
        hps.bert, hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net_g.load_state_dict(ckpt["model"])
    return net_g.eval(), hps.data.sampling_rate, ckpt.get("iteration")


@dataclass
class StepResult:
    name: str
    seconds: float
    data: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    raw: str
    steps: list[StepResult] = field(default_factory=list)
    error: str | None = None
    gt_audio: tuple[int, np.ndarray] | None = None
    mms_audio: tuple[int, np.ndarray] | None = None
    vits_audio: tuple[int, np.ndarray] | None = None


class DebugPipeline:
    def __init__(
        self,
        metadata_csv: Path | None,
        wav_dir: Path | None,
        vphon_dir: Path = Path("vphon"),
        checkpoint: Path | None = None,
    ):
        self.g2p = Text2PhonemeSequence(language="vie-n", is_cuda=False)
        self.vphon_dir = vphon_dir
        self.backends: dict[str, object] = {}
        self.xpb_tokenizer = load_xphonebert_tokenizer()
        self.xpb_model = AutoModel.from_pretrained(XPHONEBERT).eval()
        self.vocab = set(self.xpb_tokenizer.get_vocab())
        self.mms_tokenizer = AutoTokenizer.from_pretrained(MMS)
        self.mms_model = VitsModel.from_pretrained(MMS).eval()
        self.mms_vocab = set(self.mms_tokenizer.get_vocab())
        self.checkpoint = checkpoint
        self.vits = load_vits(checkpoint) if checkpoint else None

        # Normalized text -> (wav id, original text), to play the recording when the input is a dataset sentence.
        self.dataset: dict[str, tuple[str, str]] = {}
        self.wav_dir = wav_dir
        if metadata_csv:
            for line in metadata_csv.read_text(encoding="utf-8").splitlines():
                if "|" in line:
                    wav, text = line.split("|", 1)
                    self.dataset[normalize(text)] = (wav, text)

    @property
    def dict_size(self) -> int:
        return len(self.g2p.phone_dict)

    def byt5_only(self, word: str) -> str:
        """CharsiuG2P ByT5 output for one word, skipping the dictionary (same calls as infer_sentence's fallback)."""
        g = self.g2p
        inputs = g.tokenizer(f"<{g.language}>: {word}", add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            preds = g.model.generate(**inputs, num_beams=1, max_length=g.phoneme_length[g.language + ".tsv"])
        raw = g.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)[0]
        try:
            return g.segment_tool(raw, ipa=True)
        except Exception:
            return g.segment_tool(raw)

    def _charsiu_g2p(self, text: str) -> tuple[str, list[dict]]:
        phonemes = self.g2p.infer_sentence(text)
        words = text.split()
        chunks = split_words(phonemes)
        aligned = len(chunks) == len(words)
        rows = []
        for i, word in enumerate(words):
            if word in PUNCTUATION_TOKENS:
                source = "punctuation"
            elif word.replace("_", " ") in self.g2p.phone_dict:
                source = "dictionary"
            else:
                source = "ByT5 model"
            chunk = " ".join(chunks[i]) if aligned else ""
            compare = "" if source == "punctuation" else self.byt5_only(word.replace("_", " "))
            rows.append(self._row(word, source, chunk, chunk, compare))
        return phonemes, rows

    def _backend_g2p(self, text: str, name: str) -> tuple[str, list[dict]]:
        if name not in self.backends:
            self.backends[name] = g2p_backends.load(name, self.vphon_dir)
        backend = self.backends[name]
        word_phonemes, rows = [], []
        for word in text.split():
            charsiu = self.g2p.infer_sentence(word)
            if word in PUNCTUATION_TOKENS:
                rows.append(self._row(word, "punctuation", word, word, ""))
                word_phonemes.append(word)
                continue
            syllables = [s for s in word.split("-") if s]
            raws = [backend.syllable(s) for s in syllables]
            if None in raws:
                # The tool cannot read this word (English, names); fall back so the later steps still run.
                source = f"CharsiuG2P fallback ({name} cannot read it)"
                raw_shown = " | ".join(r or f"[{s}]" for s, r in zip(syllables, raws, strict=True))
                phonemes = charsiu
            else:
                source = name
                raw_shown = " | ".join(raws)
                phonemes = " - ".join(" ".join(backend.to_tokens(r)) for r in raws)
            rows.append(self._row(word, source, raw_shown, phonemes, charsiu))
            word_phonemes.append(phonemes)
        return " ▁ ".join(word_phonemes), rows

    def _row(self, word: str, source: str, raw: str, phonemes: str, compare: str) -> dict:
        report = validate(phonemes, self.vocab) if phonemes else None
        return {
            "word": word,
            "source": source,
            "raw": raw,
            "phonemes": phonemes,
            "tones": ", ".join(tone_names(phonemes)),
            "compare": compare,
            "syllables_in": 1 if word in PUNCTUATION_TOKENS else word.count("-") + 1,
            "syllables_out": len(split_syllables(phonemes.split())),
            "problems": (report.errors + report.warnings) if report else [],
        }

    def run(self, raw: str, g2p: str = g2p_backends.CHARSIU) -> PipelineResult:
        result = PipelineResult(raw=raw)
        t = time.perf_counter()

        def step(name: str, **data) -> None:
            nonlocal t
            now = time.perf_counter()
            result.steps.append(StepResult(name, now - t, data))
            t = now

        try:
            norm = normalize_with_info(raw)
        except ValueError as e:
            result.error = str(e)
            return result
        step("normalize", steps=norm.steps, dropped=norm.dropped, text=norm.text)

        try:
            if g2p == g2p_backends.CHARSIU:
                phonemes, rows = self._charsiu_g2p(norm.text)
            else:
                phonemes, rows = self._backend_g2p(norm.text, g2p)
        except Exception as e:  # a backend that fails to load should not kill the app
            result.error = f"{g2p} failed: {type(e).__name__}: {e}"
            return result
        aligned = len(split_words(phonemes)) == len(norm.text.split())
        compare_label = "ByT5 only" if g2p == g2p_backends.CHARSIU else "CharsiuG2P"
        step("g2p", backend=g2p, phonemes=phonemes, words=rows, aligned=aligned, compare_label=compare_label)

        report = validate(phonemes, self.vocab)
        step("validate", errors=report.errors, warnings=report.warnings)

        enc = self.xpb_tokenizer(phonemes, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        tokens = self.xpb_tokenizer.convert_ids_to_tokens(ids)
        unk = self.xpb_tokenizer.unk_token_id
        step(
            "xphonebert_tokens",
            tokens=list(zip(tokens, ids, strict=True)),
            count=len(ids),
            unk_positions=[i for i, x in enumerate(ids) if x == unk],
            # One id per phoneme plus <s> and </s>; anything else means the tokenizer split or dropped tokens.
            expected=len(phonemes.split()) + 2,
        )

        with torch.no_grad():
            hidden = self.xpb_model(**enc).last_hidden_state
        step(
            "xphonebert_encoder",
            shape=list(hidden.shape),
            mean=round(hidden.mean().item(), 4),
            std=round(hidden.std().item(), 4),
        )

        if self.vits:
            net_g, vits_sr, epoch = self.vits
            torch.manual_seed(0)
            with torch.no_grad():
                audio = net_g.infer(
                    enc["input_ids"], enc["attention_mask"], noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                )[0]
            vits_wave = audio[0, 0].numpy()
            result.vits_audio = (vits_sr, vits_wave)
            step(
                "vits",
                checkpoint=self.checkpoint.name,
                epoch=epoch,
                sampling_rate=vits_sr,
                seconds_of_audio=round(len(vits_wave) / vits_sr, 2),
                peak=round(float(np.abs(vits_wave).max()), 3),
            )

        mms_text = norm.text.replace("-", " ")
        mms_dropped = sorted({ch for ch in mms_text if ch not in self.mms_vocab})
        mms_inputs = self.mms_tokenizer(mms_text, return_tensors="pt")
        mms_ids = mms_inputs["input_ids"][0].tolist()
        # With add_blank the tokenizer interleaves blank id 0 between characters; show only the characters.
        if getattr(self.mms_tokenizer, "add_blank", False):
            mms_ids = mms_ids[1::2]
        mms_tokens = self.mms_tokenizer.convert_ids_to_tokens(mms_ids)
        torch.manual_seed(0)
        with torch.no_grad():
            wave = self.mms_model(**mms_inputs).waveform[0].numpy()
        sr = self.mms_model.config.sampling_rate
        result.mms_audio = (sr, wave)
        step(
            "mms_tts",
            input_text=mms_text,
            dropped=mms_dropped,
            tokens="".join(mms_tokens),
            sampling_rate=sr,
            seconds_of_audio=round(len(wave) / sr, 2),
        )

        match = self.dataset.get(norm.text)
        if match and self.wav_dir:
            audio, gt_sr = sf.read(self.wav_dir / match[0], dtype="float32")
            result.gt_audio = (gt_sr, audio)
        step("ground_truth", wav=match[0] if match else None, original=match[1] if match else None)
        return result
