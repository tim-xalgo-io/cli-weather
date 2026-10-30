"""Textual UI: one 1 km grid cell per character column, two rows per line via '▀'."""

from functools import lru_cache

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Footer, Header, Static

from . import basemap as basemap_mod
from . import grid, palette
from .source import Box, Frame, Source

# Colour-index table used by the view arrays
_TABLE = [c for _, c in palette.RAIN_STEPS] + [palette.LAND, palette.WATER, palette.NO_DATA, palette.NO_DATA_ALT]
I_LAND, I_WATER, I_NODATA, I_NODATA_ALT = range(len(palette.RAIN_STEPS), len(_TABLE))

MARKER_FG = (255, 255, 255)
LABEL_FG = (225, 225, 210)
PLAY_INTERVAL = 0.25
LOOP_HOLD_TICKS = 4
REFRESH_SECONDS = 300


@lru_cache(maxsize=512)
def _style(fg: int, bg: int) -> Style:
    return Style(color=Color.from_rgb(*_TABLE[fg]), bgcolor=Color.from_rgb(*_TABLE[bg]))


@lru_cache(maxsize=512)
def _text_style(fg: tuple[int, int, int], bg: int, bold: bool) -> Style:
    r, g, b = _TABLE[bg]
    if 0.299 * r + 0.587 * g + 0.114 * b > 140:  # bright rain colour: switch to dark text
        fg = (10, 10, 20)
    return Style(color=Color.from_rgb(*fg), bgcolor=Color.from_rgb(r, g, b), bold=bold)


def colour_indices(rain: np.ndarray | None, land: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Per-cell index into _TABLE for a viewport window."""
    base = np.where(land, I_LAND, I_WATER)
    if rain is None:
        return np.where((rows[:, None] // 2 + cols[None, :]) % 2 == 0, I_NODATA, I_NODATA_ALT)
    step = palette.rain_index(rain)
    checker = (rows[:, None] // 2 + cols[None, :]) % 2 == 0
    nodata = np.where(checker, I_NODATA, I_NODATA_ALT)
    return np.where(step >= 0, step, np.where(step == -2, nodata, base))


class MapView(Widget):
    DEFAULT_CSS = "MapView { height: 1fr; }"

    frame: reactive[Frame | None] = reactive(None)
    pan: reactive[tuple[int, int]] = reactive((0, 0))
    labels: reactive[bool] = reactive(True)
    message: reactive[str] = reactive("Loading radar…")

    def __init__(self, box: Box, base: basemap_mod.Basemap | None, **kw) -> None:
        super().__init__(**kw)
        self.box = box
        self.base = base
        self._cache_key = None
        self._view: np.ndarray | None = None
        self._overlay: dict[tuple[int, int], tuple[str, tuple[int, int, int], bool]] = {}

    def clamp(self, pan: tuple[int, int]) -> tuple[int, int]:
        """Keep the window inside the fetched box (centred on it if the window is larger)."""
        h, w = self.size.height * 2, self.size.width
        b = self.box
        out = []
        for d, centre, size, start, extent in (
            (pan[0], grid.STOCKHOLM[0], h, b.row0, b.rows),
            (pan[1], grid.STOCKHOLM[1], w, b.col0, b.cols),
        ):
            lo = start - centre + size // 2  # pan where the window's first cell is the box's first
            if size >= extent:
                out.append(lo - (size - extent) // 2)
            else:
                out.append(max(lo, min(d, lo + extent - size)))
        return out[0], out[1]

    def window(self) -> tuple[int, int, int, int]:
        """Grid window (row0, col0, rows, cols) for the current size and pan."""
        w, h = self.size.width, self.size.height * 2
        cr, cc = grid.STOCKHOLM
        dr, dc = self.clamp(self.pan)
        return cr + dr - h // 2, cc + dc - w // 2, h, w

    def _compute(self) -> None:
        key = (id(self.frame), self.pan, self.size, self.labels)
        if key == self._cache_key:
            return
        self._cache_key = key
        row0, col0, h, w = self.window()
        rows = np.arange(row0, row0 + h)
        cols = np.arange(col0, col0 + w)
        land = self.base.window(row0, col0, h, w) if self.base else np.ones((h, w), bool)

        rain = None
        if self.frame is not None and self.frame.grid is not None:
            rain = np.zeros((h, w), np.float32)  # outside the fetched box: plain basemap
            b = self.box
            r0, r1 = max(row0, b.row0), min(row0 + h, b.row0 + b.rows)
            c0, c1 = max(col0, b.col0), min(col0 + w, b.col0 + b.cols)
            if r0 < r1 and c0 < c1:
                rain[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = self.frame.grid[
                    r0 - b.row0:r1 - b.row0, c0 - b.col0:c1 - b.col0
                ]
        self._view = colour_indices(rain, land, rows, cols)

        self._overlay = {}
        places = grid.PLACES.items() if self.labels else [("Stockholm", grid.PLACES["Stockholm"])]
        for name, (lat, lon) in places:
            r, c = grid.latlon_to_rc(lat, lon)
            y, x = (r - row0) // 2, c - col0
            if not (0 <= y < h // 2 and 0 <= x < w):
                continue
            is_home = name == "Stockholm"
            self._overlay[(y, x)] = ("+" if is_home else "•", MARKER_FG, True)
            if self.labels:
                for i, ch in enumerate(" " + name):
                    if x + 1 + i < w and (y, x + 1 + i) not in self._overlay:
                        self._overlay[(y, x + 1 + i)] = (ch, LABEL_FG, is_home)

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if self.frame is None:
            line = self.message.center(width) if y == self.size.height // 2 else ""
            return Strip([Segment(line.ljust(width))])
        self._compute()
        top, bot = self._view[2 * y], self._view[2 * y + 1]
        segments: list[Segment] = []
        x = 0
        while x < width:
            over = self._overlay.get((y, x))
            if over:
                ch, fg, bold = over
                segments.append(Segment(ch, _text_style(fg, int(top[x]), bold)))
                x += 1
                continue
            start, t, b = x, top[x], bot[x]
            x += 1
            while x < width and top[x] == t and bot[x] == b and (y, x) not in self._overlay:
                x += 1
            segments.append(Segment("▀" * (x - start), _style(int(t), int(b))))
        return Strip(segments, width)

    def watch_frame(self) -> None:
        self.refresh()

    def watch_pan(self) -> None:
        self.refresh()

    def watch_labels(self) -> None:
        self.refresh()

    def watch_message(self) -> None:
        self.refresh()


class Timeline(Static):
    DEFAULT_CSS = "Timeline { height: 2; padding: 0 1; }"

    def show(self, frames: list[Frame], index: int, playing: bool) -> None:
        if not frames:
            return
        f = frames[index]
        line = Text()
        line.append("▶ " if playing else "⏸ ", style="bold")
        for i, fr in enumerate(frames):
            ch = "┃" if fr.offset == 0 else ("•" if fr.is_forecast else "·")
            style = "grey50" if fr.grid is None else ("cyan" if fr.is_forecast else "white")
            if i == index:
                style = "bold reverse " + ("cyan" if fr.is_forecast else "white")
            line.append(ch, style=style)
        local = f.time.astimezone().strftime("%H:%M")
        kind = "forecast" if f.is_forecast else ("now" if f.offset == 0 else "radar")
        line.append(f"  {local}  {f.offset:+d} min ({kind})", style="bold")
        if f.grid is None:
            line.append("  no data", style="bold red")

        legend = Text("mm/h ")
        for label, rgb in palette.legend():
            legend.append("██", style=Style(color=Color.from_rgb(*rgb)))
            legend.append(f"{label} ")
        legend.append("   Data: MET Norway (CC BY 4.0)", style="dim")
        self.update(Text("\n").join([line, legend]))


class WeatherApp(App):
    TITLE = "Stockholm precipitation"
    SUB_TITLE = "−90 … +90 min"

    BINDINGS = [
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("left", "step(-1)", "−5 min"),
        Binding("right", "step(1)", "+5 min"),
        Binding("n", "now", "Now"),
        Binding("h,shift+left", "pan(0,-1)", "Pan", show=False),
        Binding("l,shift+right", "pan(0,1)", "Pan", show=False),
        Binding("k,shift+up", "pan(-1,0)", "Pan", show=False),
        Binding("j,shift+down", "pan(1,0)", "Pan", show=False),
        Binding("H", "pan(0,-10)", "Pan", show=False),
        Binding("L", "pan(0,10)", "Pan", show=False),
        Binding("K", "pan(-10,0)", "Pan", show=False),
        Binding("J", "pan(10,0)", "Pan", show=False),
        Binding("c", "center", "Center"),
        Binding("g", "toggle_labels", "Labels"),
        Binding("r", "refresh_data", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, source: Source | None = None, live_refresh: bool = True) -> None:
        super().__init__()
        self.source = source or Source()
        self.live_refresh = live_refresh
        self.frames: list[Frame] = []
        self.index = 0
        self.playing = False
        self._hold = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield MapView(self.source.box, basemap_mod.load(), id="map")
        yield Timeline(id="timeline")
        yield Footer()

    @property
    def map(self) -> MapView:
        return self.query_one(MapView)

    def on_mount(self) -> None:
        self.load_data()
        self.set_interval(PLAY_INTERVAL, self._tick)
        if self.live_refresh:
            self.set_interval(REFRESH_SECONDS, self.load_data)

    async def on_unmount(self) -> None:
        await self.source.aclose()

    def load_data(self) -> None:
        self.run_worker(self._load(), exclusive=True, group="load")

    async def _load(self) -> None:
        keep_offset = self.frames[self.index].offset if self.frames else 0
        if not self.frames:
            self.map.message = "Loading radar…"
        try:
            frames = await self.source.load_frames()
        except Exception as e:  # network/server trouble: keep old frames if any
            if not self.frames:
                self.map.message = f"Could not load radar data: {e}"
            self.notify(f"Radar load failed: {e}", severity="error")
            return
        self.frames = frames
        offsets = [f.offset for f in frames]
        self.index = offsets.index(keep_offset) if keep_offset in offsets else offsets.index(0)
        self._show()

    def _show(self) -> None:
        if not self.frames:
            return
        self.map.frame = self.frames[self.index]
        self.query_one(Timeline).show(self.frames, self.index, self.playing)

    def _tick(self) -> None:
        if not self.playing or not self.frames:
            return
        if self.index == len(self.frames) - 1:
            self._hold += 1
            if self._hold < LOOP_HOLD_TICKS:
                return
            self._hold = 0
            self.index = 0
        else:
            self.index += 1
        self._show()

    def action_toggle_play(self) -> None:
        self.playing = not self.playing
        self._hold = 0
        if self.playing and self.frames and self.index == len(self.frames) - 1:
            self.index = 0
        self._show()

    def action_step(self, delta: int) -> None:
        if self.frames:
            self.playing = False
            self.index = max(0, min(len(self.frames) - 1, self.index + delta))
            self._show()

    def action_now(self) -> None:
        if self.frames:
            self.playing = False
            self.index = [f.offset for f in self.frames].index(0)
            self._show()

    def action_pan(self, dr: int, dc: int) -> None:
        r, c = self.map.clamp(self.map.pan)
        self.map.pan = self.map.clamp((r + dr, c + dc))

    def action_center(self) -> None:
        self.map.pan = (0, 0)

    def action_toggle_labels(self) -> None:
        self.map.labels = not self.map.labels

    def action_refresh_data(self) -> None:
        self.notify("Refreshing radar…")
        self.load_data()
