"""MET Norway radar nowcast via THREDDS OPeNDAP.

Each run file holds lwe_precipitation_rate (mm/h) for 24 five-minute steps,
step 0 being the radar analysis at the run time. Past frames are step 0 of
earlier runs; forecast frames are steps 1..18 of the newest run.
"""

import asyncio
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from . import grid

BASE = "https://thredds.met.no/thredds"
CATALOG = f"{BASE}/catalog/radarnowcasting/catalog.xml"
VAR = "lwe_precipitation_rate"
# MET Norway asks clients to identify themselves with a contact address or URL
USER_AGENT = " ".join(filter(None, ["cli-weather/0.1", os.environ.get("CLI_WEATHER_CONTACT")]))

STEP_MIN = 5
SPAN_STEPS = 18  # 90 min each way
FILL_THRESHOLD = 1e30

_RUN_RE = re.compile(r'urlPath="(radarnowcasting/[^"]*pcappi-0-rr[^"]*\.(\d{8}T\d{6})Z\.nc)"')


@dataclass(frozen=True)
class Box:
    """Inclusive-exclusive grid window [row0, row0+rows) × [col0, col0+cols)."""

    row0: int
    col0: int
    rows: int
    cols: int

    @classmethod
    def around(cls, center: tuple[int, int], rows: int, cols: int) -> "Box":
        r, c = center
        return cls(r - rows // 2, c - cols // 2, rows, cols)

    def dap(self) -> str:
        return f"[{self.row0}:{self.row0 + self.rows - 1}][{self.col0}:{self.col0 + self.cols - 1}]"


DEFAULT_BOX = Box.around(grid.STOCKHOLM, rows=160, cols=240)


@dataclass(eq=False)  # identity equality: grids are arrays, and Textual reactives compare values
class Frame:
    time: datetime
    offset: int  # minutes relative to "now" (newest run time)
    grid: np.ndarray | None  # None = data missing

    @property
    def is_forecast(self) -> bool:
        return self.offset > 0


@dataclass(frozen=True)
class Run:
    path: str
    time: datetime


def parse_catalog(xml: str) -> list[Run]:
    runs = {
        m.group(1): datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        for m in _RUN_RE.finditer(xml)
    }
    return sorted((Run(p, t) for p, t in runs.items()), key=lambda r: r.time)


def parse_dods(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode a DAP2 binary response holding a single Float32 array."""
    marker = b"\nData:\n"
    i = data.find(marker)
    if i < 0:
        raise ValueError(f"not a DAP2 response: {data[:200]!r}")
    body = data[i + len(marker):]
    n, n2 = struct.unpack(">ii", body[:8])
    expected = int(np.prod(shape))
    if n != expected or n2 != expected:
        raise ValueError(f"expected {expected} values, got {n}/{n2}")
    arr = np.frombuffer(body[8:8 + 4 * n], dtype=">f4").astype(np.float32).reshape(shape)
    arr[arr >= FILL_THRESHOLD] = np.nan
    return arr


def dods_url(path: str, t0: int, t1: int, box: Box) -> str:
    q = f"{VAR}.{VAR}[{t0}:{t1}]{box.dap()}"
    return f"{BASE}/dodsC/{path}.dods?" + q.replace("[", "%5B").replace("]", "%5D")


class Source:
    def __init__(self, box: Box = DEFAULT_BOX, client: httpx.AsyncClient | None = None):
        self.box = box
        self._client = client
        self._sem = asyncio.Semaphore(4)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def list_runs(self) -> list[Run]:
        r = await self._http().get(CATALOG)
        r.raise_for_status()
        return parse_catalog(r.text)

    async def fetch(self, path: str, t0: int, t1: int) -> np.ndarray:
        async with self._sem:
            r = await self._http().get(dods_url(path, t0, t1, self.box))
        r.raise_for_status()
        return parse_dods(r.content, (t1 - t0 + 1, self.box.rows, self.box.cols))

    async def load_frames(self) -> list[Frame]:
        """37 frames: -90 … +90 minutes around the newest run."""
        runs = await self.list_runs()
        if not runs:
            raise RuntimeError("no radar nowcast runs in THREDDS catalog")
        latest = runs[-1]
        by_time = {r.time: r for r in runs}
        past_times = [latest.time - timedelta(minutes=STEP_MIN * k) for k in range(SPAN_STEPS, -1, -1)]

        async def past(t: datetime) -> np.ndarray | None:
            run = by_time.get(t)
            if run is None:
                return None
            try:
                return (await self.fetch(run.path, 0, 0))[0]
            except (httpx.HTTPError, ValueError):
                return None

        forecast_task = self.fetch(latest.path, 1, SPAN_STEPS)
        *past_grids, forecast = await asyncio.gather(*(past(t) for t in past_times), forecast_task)

        frames = [
            Frame(t, -STEP_MIN * k, g)
            for k, t, g in zip(range(SPAN_STEPS, -1, -1), past_times, past_grids)
        ]
        frames += [
            Frame(latest.time + timedelta(minutes=STEP_MIN * s), STEP_MIN * s, forecast[s - 1])
            for s in range(1, SPAN_STEPS + 1)
        ]
        return frames
