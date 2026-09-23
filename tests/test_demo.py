import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from demo.backend import app as backend


def test_recorded_demo_starts_without_loading_models(monkeypatch):
    monkeypatch.delenv("VQA_LIVE_MODELS", raising=False)

    def unexpected_load():
        raise AssertionError("The recorded demo must not load GPU models")

    monkeypatch.setattr(backend.manager, "load_all", unexpected_load)
    with TestClient(backend.app) as client:
        assert client.get("/api/results").status_code == 200
        assert client.get("/api/gallery").status_code == 200
        response = client.post(
            "/api/ask",
            data={"question": "What?"},
            files={"image": ("image.png", b"unused", "image/png")},
        )
        assert response.status_code == 503


def test_generation_error_closes_the_event_stream(monkeypatch):
    monkeypatch.delenv("VQA_LIVE_MODELS", raising=False)
    monkeypatch.setattr(backend.manager, "specs", lambda: [{"key": "zero_shot", "loaded": True}])

    def fail(*args, **kwargs):
        raise RuntimeError("test generation failure")

    monkeypatch.setattr(backend.manager, "stream_sync", fail)
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/ask",
            data={"question": "What?", "models": "zero_shot"},
            files={"image": ("image.png", buffer.getvalue(), "image/png")},
        )
        assert '"error"' in response.text
        assert "[DONE]" in response.text


def test_black_image_condition_preserves_question_and_dimensions(monkeypatch):
    monkeypatch.delenv("VQA_LIVE_MODELS", raising=False)
    monkeypatch.setattr(backend.manager, "specs", lambda: [{"key": "zero_shot", "loaded": True}])
    received = []

    def capture(key, image, question, *args):
        received.append((key, image.size, image.getextrema(), question))

    monkeypatch.setattr(backend.manager, "stream_sync", capture)
    buffer = io.BytesIO()
    Image.new("RGB", (9, 7), "red").save(buffer, format="PNG")
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/ask",
            data={"question": "What color?", "models": "zero_shot", "image_condition": "black"},
            files={"image": ("image.png", buffer.getvalue(), "image/png")},
        )
        assert "[DONE]" in response.text
    assert received == [("zero_shot", (9, 7), ((0, 0), (0, 0), (0, 0)), "What color?")]
