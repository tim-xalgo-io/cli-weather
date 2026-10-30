"""Pre-rasterised land/water mask on the radar grid (built by tools/build_basemap.py)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CANDIDATES = [
    Path(__file__).resolve().parent / "data" / "basemap.npz",  # installed wheel
    Path(__file__).resolve().parents[2] / "data" / "basemap.npz",  # source checkout
]


@dataclass
class Basemap:
    land: np.ndarray  # uint8 [rows, cols], 1 = land
    row0: int
    col0: int

    def window(self, row0: int, col0: int, rows: int, cols: int) -> np.ndarray:
        """Land mask for a grid window; cells outside the basemap count as land."""
        out = np.ones((rows, cols), dtype=bool)
        r0, c0 = row0 - self.row0, col0 - self.col0
        h, w = self.land.shape
        sr0, sr1 = max(r0, 0), min(r0 + rows, h)
        sc0, sc1 = max(c0, 0), min(c0 + cols, w)
        if sr0 < sr1 and sc0 < sc1:
            out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = self.land[sr0:sr1, sc0:sc1].astype(bool)
        return out


def load() -> Basemap | None:
    for path in _CANDIDATES:
        if path.exists():
            d = np.load(path)
            row0, col0 = (int(v) for v in d["origin"])
            return Basemap(d["land"], row0, col0)
    return None
