"""Colours: yr-like precipitation steps over a muted land/water basemap."""

import numpy as np

RGB = tuple[int, int, int]

RAIN_MIN = 0.1  # mm/h; below this the basemap shows through

# (lower bound mm/h, colour)
RAIN_STEPS: list[tuple[float, RGB]] = [
    (0.1, (150, 210, 255)),
    (0.5, (90, 170, 245)),
    (1.0, (40, 120, 230)),
    (2.0, (20, 70, 200)),
    (5.0, (120, 50, 200)),
    (10.0, (200, 40, 160)),
]

LAND: RGB = (58, 66, 54)
WATER: RGB = (28, 36, 52)
NO_DATA: RGB = (70, 70, 70)
NO_DATA_ALT: RGB = (52, 52, 52)

_BOUNDS = np.array([b for b, _ in RAIN_STEPS])
_COLOURS = [c for _, c in RAIN_STEPS]


def rain_colour(mm_h: float) -> RGB | None:
    """Colour for a rain rate, or None if dry (or NaN)."""
    if not mm_h >= RAIN_MIN:
        return None
    return _COLOURS[int(np.searchsorted(_BOUNDS, mm_h, side="right")) - 1]


def rain_index(grid: np.ndarray) -> np.ndarray:
    """Vectorised step index per cell: -1 dry, -2 no data, else index into RAIN_STEPS."""
    idx = np.searchsorted(_BOUNDS, np.nan_to_num(grid, nan=0.0), side="right") - 1
    idx[np.isnan(grid)] = -2
    return idx


def cell_colour(step: int, land: bool, checker: bool) -> RGB:
    if step >= 0:
        return _COLOURS[step]
    if step == -2:
        return NO_DATA if checker else NO_DATA_ALT
    return LAND if land else WATER


def legend() -> list[tuple[str, RGB]]:
    labels = ["0.1", "0.5", "1", "2", "5", "10+"]
    return list(zip(labels, _COLOURS))
