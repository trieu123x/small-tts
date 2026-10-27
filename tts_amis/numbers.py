"""Read non-negative integers as Vietnamese words (northern reading)."""

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
# Group names for 10^3, 10^6, 10^9.
SCALES = ["", "nghìn", "triệu", "tỷ"]
MAX_GROUPED = 10**12 - 1


def _tens_and_units(tens: int, units: int) -> list[str]:
    if tens == 0:
        return [DIGITS[units]] if units else []
    words = ["mười"] if tens == 1 else [DIGITS[tens], "mươi"]
    if units == 1 and tens > 1:
        words.append("mốt")
    elif units == 4 and tens > 1:
        words.append("tư")
    elif units == 5:
        words.append("lăm")
    elif units:
        words.append(DIGITS[units])
    return words


def _group(n: int, full: bool) -> list[str]:
    """Read 0..999. `full` forces "không trăm" / "linh" when a higher group precedes this one."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words: list[str] = []
    if hundreds or full:
        words += [DIGITS[hundreds], "trăm"]
    if tens == 0 and units and words:
        words += ["linh", DIGITS[units]]
    else:
        words += _tens_and_units(tens, units)
    return words


def to_words(n: int) -> str:
    """`21 -> "hai mươi mốt"`, `105 -> "một trăm linh năm"`. Numbers past 10^12 are read digit by digit."""
    if n < 0:
        raise ValueError(f"negative numbers are not supported: {n}")
    if n == 0:
        return DIGITS[0]
    if n > MAX_GROUPED:
        return " ".join(DIGITS[int(d)] for d in str(n))

    groups = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)

    words: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        words += _group(g, full=bool(words))
        if SCALES[i]:
            words.append(SCALES[i])
    return " ".join(words)
