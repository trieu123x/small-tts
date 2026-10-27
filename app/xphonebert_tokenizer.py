"""The one correct way to load the XPhoneBERT tokenizer. Shared by the debug app and the vendored VITS training code."""

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

XPHONEBERT = "vinai/xphonebert-base"


def load_xphonebert_tokenizer(repo: str = XPHONEBERT) -> PreTrainedTokenizerFast:
    """Load the WordLevel + WhitespaceSplit tokenizer from tokenizer.json.

    AutoTokenizer in transformers 5.x builds a byte-level BPE RobertaTokenizer for this repo instead, which silently
    drops every non-ASCII phoneme (tones, `▁`, `ʔ`, `ɤ̆`).
    """
    return PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(repo, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
