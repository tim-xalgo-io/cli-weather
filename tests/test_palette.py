import numpy as np

from cli_weather import palette


def test_rain_colour_bounds():
    assert palette.rain_colour(0.0) is None
    assert palette.rain_colour(0.099) is None
    assert palette.rain_colour(float("nan")) is None
    assert palette.rain_colour(0.1) == palette.RAIN_STEPS[0][1]
    assert palette.rain_colour(0.49) == palette.RAIN_STEPS[0][1]
    assert palette.rain_colour(0.5) == palette.RAIN_STEPS[1][1]
    assert palette.rain_colour(50.0) == palette.RAIN_STEPS[-1][1]


def test_rain_index():
    idx = palette.rain_index(np.array([0.0, 0.1, 1.5, 12.0, np.nan]))
    assert idx.tolist() == [-1, 0, 2, 5, -2]
