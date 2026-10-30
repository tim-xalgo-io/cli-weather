from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cli_weather.source import Box, dods_url, parse_catalog, parse_dods

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_dods_matches_ascii():
    arr = parse_dods((FIXTURES / "small.dods").read_bytes(), (2, 3, 4))
    expected = [
        [float(v) for v in line.split(", ")[1:]]
        for line in (FIXTURES / "small.ascii.txt").read_text().splitlines()
    ]
    np.testing.assert_allclose(arr.reshape(6, 4), expected, rtol=1e-6)


def test_parse_dods_fill_is_nan():
    arr = parse_dods((FIXTURES / "fill.dods").read_bytes(), (1, 2, 2))
    assert np.isnan(arr).all()


def test_parse_dods_shape_mismatch():
    with pytest.raises(ValueError):
        parse_dods((FIXTURES / "small.dods").read_bytes(), (1, 3, 4))


def test_parse_catalog_sorted_and_filtered():
    xml = """
      <dataset urlPath="radarnowcasting/yrwms-nordic.mos.pcappi-0-rr.x.nordiclcc-1000.20261008T130500Z.nc"/>
      <dataset urlPath="radarnowcasting/yrwms-nordic.mos.pcappi-0-dbz.x.nordiclcc-1000.20261008T130000Z.nc"/>
      <dataset urlPath="radarnowcasting/yrwms-nordic.mos.pcappi-0-rr.x.nordiclcc-1000.20261008T130000Z.nc"/>
    """
    runs = parse_catalog(xml)
    assert [r.time for r in runs] == [
        datetime(2026, 10, 8, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 8, 13, 5, tzinfo=timezone.utc),
    ]


def test_dods_url_encodes_brackets():
    url = dods_url("radarnowcasting/f.nc", 1, 18, Box(10, 20, 3, 4))
    assert url.endswith("f.nc.dods?lwe_precipitation_rate.lwe_precipitation_rate%5B1:18%5D%5B10:12%5D%5B20:23%5D")
