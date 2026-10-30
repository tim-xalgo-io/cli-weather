from datetime import datetime, timedelta, timezone

import numpy as np

from cli_weather import grid
from cli_weather.app import WeatherApp
from cli_weather.source import DEFAULT_BOX, Frame

NOW = datetime(2026, 10, 8, 13, 0, tzinfo=timezone.utc)


class FakeSource:
    box = DEFAULT_BOX

    def __init__(self, fail: bool = False):
        self.fail = fail

    async def load_frames(self):
        if self.fail:
            raise RuntimeError("offline")
        frames = []
        for k in range(-18, 19):
            g = np.zeros((self.box.rows, self.box.cols), np.float32)
            g[:, : 40 + k] = 3.0  # a rain band moving east
            frames.append(Frame(NOW + timedelta(minutes=5 * k), 5 * k, None if k == -10 else g))
        return frames

    async def aclose(self):
        pass


async def test_navigation_and_render():
    app = WeatherApp(source=FakeSource(), live_refresh=False)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.frames) == 37
        assert app.frames[app.index].offset == 0

        await pilot.press("right")
        assert app.frames[app.index].offset == 5
        await pilot.press("left", "left")
        assert app.frames[app.index].offset == -5
        await pilot.press("n")
        assert app.frames[app.index].offset == 0

        await pilot.press("space")
        assert app.playing
        await pilot.press("space")
        assert not app.playing

        await pilot.press("l", "L", "j")
        assert app.map.pan == (1, 11)
        await pilot.press("c")
        assert app.map.pan == (0, 0)

        # Stockholm marker sits at the map centre
        row0, col0, h, w = app.map.window()
        r, c = grid.STOCKHOLM
        line = app.map.render_line((r - row0) // 2)
        assert line.text[c - col0] == "+"

        # missing frame renders without crashing
        app.index = [f.offset for f in app.frames].index(-50)
        app._show()
        await pilot.pause()
        assert app.map.render_line(0).cell_length == w


async def test_refresh_with_same_runs_keeps_running():
    app = WeatherApp(source=FakeSource(), live_refresh=False)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("right")
        await pilot.press("r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.is_running
        assert app.frames[app.index].offset == 5
        assert app.map.frame is app.frames[app.index]


async def test_pan_is_clamped():
    app = WeatherApp(source=FakeSource(), live_refresh=False)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.workers.wait_for_complete()
        for _ in range(30):
            await pilot.press("L")
        row0, col0, h, w = app.map.window()
        box = app.source.box
        assert col0 + w == box.col0 + box.cols


async def test_load_failure_shows_message():
    app = WeatherApp(source=FakeSource(fail=True), live_refresh=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.frames == []
        assert "offline" in app.map.message
