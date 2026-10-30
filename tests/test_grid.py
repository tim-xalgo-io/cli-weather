import pytest

from cli_weather import grid


def test_stockholm_cell():
    assert grid.latlon_to_rc(59.3293, 18.0686) == (1529, 970)
    assert grid.STOCKHOLM == (1529, 970)


@pytest.mark.parametrize(
    "rc, latlon",
    [  # reference values read from MET's own lat/lon arrays in the THREDDS file
        ((1529, 970), (59.3319, 18.063002)),
        ((0, 0), (71.78062, -8.085339)),
        ((2133, 1693), (53.139175, 28.362951)),
    ],
)
def test_matches_met_latlon(rc, latlon):
    lat, lon = grid.rc_to_latlon(*rc)
    assert lat == pytest.approx(latlon[0], abs=1e-4)
    assert lon == pytest.approx(latlon[1], abs=1e-4)


def test_roundtrip():
    for lat, lon in grid.PLACES.values():
        r, c = grid.latlon_to_rc(lat, lon)
        lat2, lon2 = grid.rc_to_latlon(r, c)
        assert abs(lat2 - lat) < 0.01 and abs(lon2 - lon) < 0.02
