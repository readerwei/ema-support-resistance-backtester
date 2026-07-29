import pandas as pd

from ema_sr.data import fetch_yahoo_bars


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "chart": {
                "result": [{
                    "timestamp": [1704110400],
                    "indicators": {"quote": [{"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [1000]}]},
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                }],
                "error": None,
            }
        }


def test_extended_session_requests_pre_and_post_market(monkeypatch):
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured.update(params)
        return FakeResponse()

    monkeypatch.setattr("ema_sr.data.requests.get", fake_get)
    result = fetch_yahoo_bars("AAPL", "2024-01-01", "2024-01-02", "1m", session="extended")
    assert captured["includePrePost"] == "true"
    assert len(result) == 1


def test_regular_session_is_default(monkeypatch):
    captured = {}
    monkeypatch.setattr("ema_sr.data.requests.get", lambda url, params, headers, timeout: (captured.update(params) or FakeResponse()))
    fetch_yahoo_bars("AAPL", "2024-01-01", "2024-01-02", "1m")
    assert captured["includePrePost"] == "false"
