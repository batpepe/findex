# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=11"]
# ///
"""Render docs/demo.gif: real `findex search` runs, typed and printed like a terminal.

    uv run scripts/demo_gif.py [index.bin] [--semantic]   (findex must be on PATH, e.g. `uv tool install .`)

Each scene types a command, runs it for real (colors and all, via FORCE_COLOR), and
shows the output. Nothing is mocked: the GIF is the program's actual output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 96, 26
FONT = "DejaVuSansMono.ttf"
SIZE = 15
BG, FG = (13, 17, 23), (201, 209, 217)
BASE = [(72, 79, 88), (255, 123, 114), (63, 185, 80), (210, 153, 34),
        (88, 166, 255), (188, 140, 255), (57, 197, 207), (177, 186, 196)]  # fmt: skip
BRIGHT = [(110, 118, 129), (255, 161, 152), (86, 211, 100), (227, 179, 65),
          (121, 192, 255), (210, 168, 255), (86, 212, 221), (240, 246, 252)]  # fmt: skip
SGR = re.compile(r"\x1b\[([0-9;]*)m")


def xterm(n: int) -> tuple[int, int, int]:
    if n < 8:
        return BASE[n]
    if n < 16:
        return BRIGHT[n - 8]
    if n < 232:
        n -= 16
        levels = [0, 95, 135, 175, 215, 255]
        return levels[n // 36], levels[n // 6 % 6], levels[n % 6]
    gray = 8 + 10 * (n - 232)
    return gray, gray, gray


type Cell = tuple[str, tuple[int, int, int], bool]


def parse(text: str) -> list[list[Cell]]:
    """ANSI text -> lines of (char, color, bold) cells."""
    lines: list[list[Cell]] = [[]]
    color, bold, dim = FG, False, False
    pos = 0
    for match in [*SGR.finditer(text), None]:
        chunk = text[pos : match.start() if match else len(text)]
        for ch in chunk:
            if ch == "\n":
                lines.append([])
            else:
                shade = tuple(c * 2 // 3 for c in color) if dim else color
                lines[-1].append((ch, (shade[0], shade[1], shade[2]), bold))
        if match is None:
            break
        pos = match.end()
        codes = [int(c) if c else 0 for c in match.group(1).split(";")]
        i = 0
        while i < len(codes):
            code = codes[i]
            if code == 0:
                color, bold, dim = FG, False, False
            elif code == 1:
                bold = True
            elif code == 2:
                dim = True
            elif code == 22:
                bold = dim = False
            elif 30 <= code <= 37:
                color = BASE[code - 30]
            elif 90 <= code <= 97:
                color = BRIGHT[code - 90]
            elif code == 39:
                color = FG
            elif code == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
                color = xterm(codes[i + 2])
                i += 2
            elif code == 38 and i + 4 < len(codes) and codes[i + 1] == 2:
                color = (codes[i + 2], codes[i + 3], codes[i + 4])
                i += 4
            i += 1
    return lines


class Screen:
    def __init__(self) -> None:
        self.font = ImageFont.truetype(FONT, SIZE)
        self.bold = ImageFont.truetype(FONT.replace(".ttf", "-Bold.ttf"), SIZE)
        ascent, descent = self.font.getmetrics()
        self.cw, self.ch = round(self.font.getlength("M")), ascent + descent  # box-drawing glyphs touch
        self.lines: list[list[Cell]] = []
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def frame(self, ms: int) -> None:
        img = Image.new("RGB", (COLS * self.cw + 24, ROWS * self.ch + 24), BG)
        draw = ImageDraw.Draw(img)
        for row, line in enumerate(self.lines[-ROWS:]):
            for col, (ch, color, bold) in enumerate(line[:COLS]):
                xy = (12 + col * self.cw, 12 + row * self.ch)
                draw.text(xy, ch, fill=color, font=self.bold if bold else self.font)
        self.frames.append(img)
        self.durations.append(ms)

    def type(self, command: str) -> None:
        self.lines.append([("$", BASE[2], True), (" ", FG, False)])
        for i in range(0, len(command), 3):
            self.lines[-1].extend((ch, FG, False) for ch in command[i : i + 3])
            self.frame(45)
        self.frame(400)

    def run(self, args: list[str], hold_ms: int) -> None:
        env = os.environ | {"FORCE_COLOR": "1", "COLUMNS": str(COLS), "TERM": "xterm-256color"}
        out = subprocess.run(args, capture_output=True, text=True, env=env, check=True).stdout
        self.lines.extend(parse(out.rstrip("\n")))
        self.frame(hold_ms)
        self.lines = []


def main() -> None:
    index = next((a for a in sys.argv[1:] if not a.startswith("--")), "index.bin")
    scenes = [
        ["findex", "search", index, "free-threaded GIL", "-k", "3"],
        ["findex", "search", index, '"event loop" NOT asyncio', "-k", "2"],
    ]
    if "--semantic" in sys.argv:  # needs `findex embed index.bin` first
        question = "find out how long a piece of code takes to run"
        scenes.append(["findex", "search", index, question, "--mode", "semantic", "-k", "2"])
    screen = Screen()
    for args in scenes:
        shown = " ".join(f"'{a}'" if " " in a else a for a in args)
        screen.type(shown)
        screen.run(args, hold_ms=3500)
    out = Path(__file__).parent.parent / "docs" / "demo.gif"
    first, rest = screen.frames[0], screen.frames[1:]
    first.save(out, save_all=True, append_images=rest, duration=screen.durations, loop=0)
    print(f"{out}: {len(screen.frames)} frames, {out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
