"""Time one VITS+XPhoneBERT training step on a single GPU under several settings (fp16, cudnn.benchmark, frozen BERT).

Each step mirrors train.py's train_and_evaluate (same losses, same order), without DDP. Prints a table of mean
seconds per phase so the 10 h run can use the fastest setting.

    python scripts/vits_profile.py --config <amis_base_xphonebert.json> --steps 30
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VITS = ROOT / "third_party" / "vits_xphonebert"
sys.path[:0] = [str(ROOT), str(VITS)]

import commons  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402
from data_utils import DistributedBucketSampler, TextAudioCollate, TextAudioLoader  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # noqa: E402
from models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.xphonebert_tokenizer import load_xphonebert_tokenizer  # noqa: E402

SETTINGS = [
    # name, fp16, cudnn.benchmark, bert frozen
    ("upstream: fp32, benchmark, bert frozen", False, True, True),
    ("fp32, no benchmark, bert frozen", False, False, True),
    ("fp16, no benchmark, bert frozen", True, False, True),
    ("fp16, benchmark, bert frozen", True, True, True),
    ("fp16, no benchmark, bert trainable", True, False, False),
]


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def profile(hps, loader, fp16: bool, benchmark: bool, frozen: bool, steps: int, warmup: int) -> dict:
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(hps.train.seed)
    d, seg = hps.data, hps.train.segment_size // hps.data.hop_length
    net_g = SynthesizerTrn(hps.bert, d.filter_length // 2 + 1, seg, **hps.model).cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
    for p in net_g.enc_p.bert.parameters():
        p.requires_grad = not frozen
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    torch.cuda.reset_peak_memory_stats()

    times: dict[str, list[float]] = {"data": [], "g_fwd": [], "d_step": [], "g_step": [], "total": []}
    it = iter(loader)
    for step in range(warmup + steps):
        t0 = sync()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, mask, _, spec, spec_len, y, _ = (b.cuda(non_blocking=True) for b in batch)
        t1 = sync()
        with torch.amp.autocast("cuda", enabled=fp16):
            y_hat, l_length, _, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(x, mask, spec, spec_len)
            mel = spec_to_mel_torch(spec, d.filter_length, d.n_mel_channels, d.sampling_rate, d.mel_fmin, d.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, seg)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), d.filter_length, d.n_mel_channels, d.sampling_rate, d.hop_length, d.win_length,
                d.mel_fmin, d.mel_fmax,
            )  # fmt: skip
            y = commons.slice_segments(y, ids_slice * d.hop_length, hps.train.segment_size)
        t2 = sync()
        with torch.amp.autocast("cuda", enabled=fp16):
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with torch.amp.autocast("cuda", enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        t3 = sync()
        with torch.amp.autocast("cuda", enabled=fp16):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with torch.amp.autocast("cuda", enabled=False):
                loss_g = (
                    generator_loss(y_d_hat_g)[0]
                    + feature_loss(fmap_r, fmap_g)
                    + F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    + torch.sum(l_length.float())
                    + kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                )
        optim_g.zero_grad()
        scaler.scale(loss_g).backward()
        scaler.unscale_(optim_g)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        t4 = sync()
        if not torch.isfinite(loss_g):
            raise RuntimeError(f"non-finite generator loss at step {step}")
        if step >= warmup:
            for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0), strict=True):
                times[k].append(v)
    result = {k: statistics.mean(v) for k, v in times.items()}
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del net_g, net_d, optim_g, optim_d
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(str(args.config))
    tokenizer = load_xphonebert_tokenizer(hps.bert)
    dataset = TextAudioLoader(hps.data.training_files, hps.data, tokenizer)
    sampler = DistributedBucketSampler(
        dataset, hps.train.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas=1, rank=0
    )
    loader = DataLoader(
        dataset, num_workers=4, pin_memory=True, batch_sampler=sampler,
        collate_fn=TextAudioCollate(pad_token_id=tokenizer.pad_token_id),
    )  # fmt: skip
    print(
        f"GPU {torch.cuda.get_device_name(0)} | batch {hps.train.batch_size} | {len(loader)} batches/epoch", flush=True
    )

    header = f"{'setting':42} {'data':>6} {'g_fwd':>6} {'d_step':>6} {'g_step':>6} {'total':>6} {'peakGB':>6}"
    rows = []
    for name, fp16, benchmark, frozen in SETTINGS:
        try:
            r = profile(hps, loader, fp16, benchmark, frozen, args.steps, args.warmup)
            row = f"{name:42} " + " ".join(f"{r[k]:6.2f}" for k in ("data", "g_fwd", "d_step", "g_step", "total"))
            row += f" {r['peak_gb']:6.1f}"
        except Exception as e:  # OOM or a numerical failure in one setting should not hide the others
            row = f"{name:42} FAILED: {type(e).__name__}: {e}"
            torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)
    print("\nseconds per step (mean of measured steps)\n" + header + "\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
