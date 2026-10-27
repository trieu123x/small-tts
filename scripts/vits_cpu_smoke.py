"""Two real training steps + one inference of the vendored VITS+XPhoneBERT on CPU, on a few utterances.

Proves the patched upstream code runs end to end (data loading, stft, tokenizer, alignment, losses, backward,
inference) before spending Kaggle GPU time. Mirrors one iteration of train.py's train_and_evaluate without DDP/AMP.

    python scripts/vits_cpu_smoke.py --filelist runs/vits-smoke/filelists/amis_train.txt --work runs/vits-smoke
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VITS = ROOT / "third_party" / "vits_xphonebert"
sys.path[:0] = [str(ROOT), str(VITS)]

import commons  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402
from data_utils import TextAudioCollate, TextAudioLoader  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # noqa: E402
from models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.xphonebert_tokenizer import load_xphonebert_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=VITS / "configs" / "amis_base_xphonebert.json")
    parser.add_argument("--n", type=int, default=4, help="utterances to use")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    # Copy a few wavs so the loader's .spec.pt cache files land in the work dir, not in the dataset.
    wav_dir = args.work / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in args.filelist.read_text(encoding="utf-8").splitlines()[: args.n]:
        src, phonemes = line.split("|", 1)
        dst = wav_dir / Path(src).name
        shutil.copy(src, dst)
        lines.append(f"{dst.resolve().as_posix()}|{phonemes}")
    mini = args.work / "mini.txt"
    mini.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hps = utils.get_hparams_from_file(str(args.config))
    torch.manual_seed(hps.train.seed)
    tokenizer = load_xphonebert_tokenizer(hps.bert)
    dataset = TextAudioLoader(str(mini), hps.data, tokenizer)
    assert len(dataset) == args.n, f"loader filtered rows: {len(dataset)} of {args.n}"
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=TextAudioCollate(pad_token_id=tokenizer.pad_token_id)
    )

    net_g = SynthesizerTrn(
        hps.bert, hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    seg = hps.train.segment_size // hps.data.hop_length
    d = hps.data

    for step, (x, mask, x_len, spec, spec_len, y, _y_len) in enumerate(loader):
        if step == args.steps:
            break
        t0 = time.time()
        # Every phoneme must survive tokenization: one id each plus <s> and </s>.
        for i in range(x.size(0)):
            assert int(mask[i].sum()) == int(x_len[i]), "attention mask / length mismatch"
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
            x, mask, spec, spec_len
        )
        mel = spec_to_mel_torch(spec, d.filter_length, d.n_mel_channels, d.sampling_rate, d.mel_fmin, d.mel_fmax)
        y_mel = commons.slice_segments(mel, ids_slice, seg)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), d.filter_length, d.n_mel_channels, d.sampling_rate, d.hop_length, d.win_length,
            d.mel_fmin, d.mel_fmax,
        )  # fmt: skip
        y = commons.slice_segments(y, ids_slice * d.hop_length, hps.train.segment_size)

        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        losses = {"disc": loss_disc, "gen": loss_gen, "fm": loss_fm, "mel": loss_mel, "dur": loss_dur, "kl": loss_kl}
        assert all(torch.isfinite(v) for v in losses.values()), losses
        shown = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
        print(f"step {step}: tokens={tuple(x.shape)} spec={tuple(spec.shape)} {shown} ({time.time() - t0:.1f}s)")

    net_g.eval()
    with torch.no_grad():
        audio = net_g.infer(x[:1], mask[:1], noise_scale=0.667, noise_scale_w=0.8, length_scale=1, max_len=400)[0]
    wave = audio[0, 0].numpy()
    out = args.work / "smoke_infer.wav"
    sf.write(out, wave, d.sampling_rate)
    print(f"inference: {len(wave) / d.sampling_rate:.2f}s of audio -> {out}")


if __name__ == "__main__":
    main()
