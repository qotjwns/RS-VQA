from __future__ import annotations

import re


def parse_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))
