"""Gradio app that shows the output of every TTS pipeline step for one sentence.

Run on a machine with torch (Kaggle): `python -m app.debug_app --data-dir <dir with metadata.csv and wavs/> --share`.
"""

import argparse
import time
from pathlib import Path

import gradio as gr

from app import g2p_backends
from app.pipeline import DebugPipeline
from app.render import render_all

VITS_NOTE = (
    "No VITS checkpoint given (`--checkpoint`), so this pipeline stops at the XPhoneBERT encoder. "
    "Audio comes from the pretrained MMS-TTS reference (generic voice, character input, no G2P)."
)
VITS_NOTE_CKPT = (
    "VITS + XPhoneBERT audio comes from checkpoint `{name}`. MMS-TTS is a pretrained reference "
    "(generic voice, character input, no G2P)."
)


def build_ui(pipe: DebugPipeline, examples: list[str]) -> gr.Blocks:
    def run(text: str, g2p: str):
        result = pipe.run(text or "", g2p)
        if result.error:
            return f"❌ {result.error}", None, None, None
        steps = [(s.name, s.seconds, s.data) for s in result.steps]
        return render_all(steps), result.vits_audio, result.mms_audio, result.gt_audio

    note = VITS_NOTE_CKPT.format(name=pipe.checkpoint.name) if pipe.checkpoint else VITS_NOTE
    with gr.Blocks(title="tts-amis pipeline debug") as demo:
        gr.Markdown(
            f"# tts-amis pipeline debug\n\nCharsiuG2P vie-n (dict: {pipe.dict_size:,} entries) → "
            f"XPhoneBERT → audio.\n\n{note}"
        )
        with gr.Row():
            text = gr.Textbox(label="Input text", lines=2, scale=4)
            pick = gr.Dropdown(choices=examples, label="…or pick a dataset sentence", scale=3)
        with gr.Row():
            g2p = gr.Radio(choices=g2p_backends.ALL, value=g2p_backends.CHARSIU, label="G2P", scale=4)
            button = gr.Button("Run pipeline", variant="primary", scale=1)
        with gr.Row():
            vits = gr.Audio(label="VITS + XPhoneBERT (checkpoint)", type="numpy", visible=pipe.vits is not None)
            mms = gr.Audio(label="MMS-TTS (pretrained reference)", type="numpy")
            gt = gr.Audio(label="Dataset recording (ground truth)", type="numpy")
        steps = gr.Markdown()

        pick.change(lambda s: s, inputs=pick, outputs=text)
        for trigger in (button.click, text.submit, g2p.change):
            trigger(run, inputs=[text, g2p], outputs=[steps, vits, mms, gt])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="folder with metadata.csv and wavs/")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = run forever)")
    parser.add_argument("--checkpoint", type=Path, help="trained VITS generator (G_*.pth) to synthesize with")
    parser.add_argument("--vphon-dir", type=Path, default=Path("vphon"), help="folder with vPhon.py and rules.py")
    args = parser.parse_args()

    metadata = args.data_dir / "metadata.csv" if args.data_dir else None
    pipe = DebugPipeline(metadata, args.data_dir / "wavs" if args.data_dir else None, args.vphon_dir, args.checkpoint)
    examples = [text for _, text in pipe.dataset.values()]
    demo = build_ui(pipe, examples)

    _, local_url, share_url = demo.launch(share=args.share, prevent_thread_lock=True)
    print(f"LOCAL_URL {local_url}", flush=True)
    print(f"SHARE_URL {share_url}", flush=True)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    while deadline is None or time.time() < deadline:
        time.sleep(30)
    demo.close()


if __name__ == "__main__":
    main()
