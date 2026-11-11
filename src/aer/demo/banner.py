"""Shared banner for every demonstration renderer."""

from typing import Literal

NOT_ADVICE = "Demonstration output of a research pipeline. Not investment advice."


def render_banner(mode: Literal["synthetic"] = "synthetic") -> str:
    if mode != "synthetic":
        raise ValueError(f"unknown banner mode: {mode}")
    lines = [
        "SYNTHETIC DATA",
        "No market data and no real company.",
        NOT_ADVICE,
        "Author: Adi Mendelowitz",
    ]
    return "\n".join(lines)
