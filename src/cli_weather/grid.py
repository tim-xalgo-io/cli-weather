"""MET Norway Nordic radar grid: 1 km Lambert conformal conic.

proj4: +proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +R=6.371e6
Xc runs -796000 → 897000 (1694 cols), Yc runs 1125000 → -1008000 (2134 rows).
"""

import math

R = 6.371e6
LAT0 = math.radians(63.0)
LON0 = math.radians(15.0)
N = math.sin(LAT0)
F = math.cos(LAT0) * math.tan(math.pi / 4 + LAT0 / 2) ** N / N

X0 = -796000.0
Y0 = 1125000.0
STEP = 1000.0
NCOLS = 1694
NROWS = 2134


def _rho(lat: float) -> float:
    return R * F / math.tan(math.pi / 4 + lat / 2) ** N


RHO0 = _rho(LAT0)


def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    rho = _rho(math.radians(lat))
    theta = N * (math.radians(lon) - LON0)
    return rho * math.sin(theta), RHO0 - rho * math.cos(theta)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    dy = RHO0 - y
    rho = math.copysign(math.hypot(x, dy), N)
    theta = math.atan2(x, dy)
    lat = 2 * math.atan((R * F / rho) ** (1 / N)) - math.pi / 2
    return math.degrees(lat), math.degrees(theta / N + LON0)


def latlon_to_rc(lat: float, lon: float) -> tuple[int, int]:
    """Grid (row, col) of the cell containing lat/lon."""
    x, y = latlon_to_xy(lat, lon)
    return round((Y0 - y) / STEP), round((x - X0) / STEP)


def rc_to_latlon(row: int, col: int) -> tuple[float, float]:
    return xy_to_latlon(X0 + col * STEP, Y0 - row * STEP)


# Sergels torg
STOCKHOLM = latlon_to_rc(59.3326, 18.0649)

PLACES = {
    "Stockholm": (59.3326, 18.0649),
    "Uppsala": (59.8586, 17.6389),
    "Södertälje": (59.1955, 17.6253),
    "Norrtälje": (59.7580, 18.7049),
    "Nynäshamn": (58.9034, 17.9479),
    "Arlanda": (59.6498, 17.9238),
    "Värmdö": (59.3160, 18.5130),
    "Enköping": (59.6356, 17.0778),
    "Västerås": (59.6099, 16.5448),
}
