"""Read the dataset's metadata.csv and write VITS filelists (`<wav path>|<phonemes>`)."""

from pathlib import Path


def read_metadata(path: Path) -> list[tuple[str, str]]:
    """Parse `N.wav|text` lines. Blank lines are skipped; a malformed line raises ValueError naming its line number."""
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        wav, sep, text = line.partition("|")
        if not sep or not wav.strip() or not text.strip():
            raise ValueError(f"{path}:{lineno}: expected 'file.wav|text', got {line!r}")
        rows.append((wav.strip(), text.strip()))
    return rows


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    for wav, phonemes in rows:
        if "|" in wav or "|" in phonemes or "\n" in phonemes:
            raise ValueError(f"field contains a separator: {wav!r}")
    path.write_text("".join(f"{wav}|{phonemes}\n" for wav, phonemes in rows), encoding="utf-8", newline="\n")
