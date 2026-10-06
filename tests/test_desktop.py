"""Desktop shell and packaged-app behaviour: model paths, engine policy, WebView2 profile, closing,
unwritable export folders and what the log keeps (invented data only)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from anonymizer import app
from anonymizer.api import server
from anonymizer.api.server import create_app
from anonymizer.engine import faces
from anonymizer.engine.fake import FakeEngine
from tests.live_client import LiveClient
from tests.test_api import TOKEN, make_pdf, upload, wait_status


@pytest.mark.skipif(not faces.available(), reason="needs the YuNet model (scripts/download_models.py)")
def test_face_model_loads_from_a_folder_with_accents(tmp_path):
    folder = tmp_path / "Año 2026 – Ñuble"
    folder.mkdir()
    model = folder / faces.YUNET_MODEL.name
    shutil.copyfile(faces.YUNET_MODEL, model)
    detector = faces.load_detector(model)
    detector.setInputSize((320, 320))
    _, detections = detector.detect(np.zeros((320, 320, 3), np.uint8))
    assert detections is None or len(detections) == 0


def test_packaged_app_refuses_the_development_engine(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("ANONYMIZER_ENGINE", "real")
    with pytest.raises(app.StartupError):
        app.load_engine("fake")


def test_packaged_app_never_falls_back_to_the_development_engine(monkeypatch):
    import anonymizer.engine

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("ANONYMIZER_ENGINE", "fake")  # ignored by the packaged app
    monkeypatch.setattr(anonymizer.engine, "get_engine", lambda: FakeEngine())  # as if the models were missing
    with pytest.raises(app.StartupError, match="antivirus"):
        app.load_engine(None)
    assert os.environ["ANONYMIZER_ENGINE"] == "real"


def test_development_app_still_falls_back(monkeypatch):
    import anonymizer.engine

    monkeypatch.setenv("ANONYMIZER_ENGINE", "real")
    monkeypatch.setattr(anonymizer.engine, "get_engine", lambda: FakeEngine())
    assert app.load_engine(None).name == "fake"


def test_each_instance_has_its_own_webview_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "app_data_dir", lambda: tmp_path)
    root = tmp_path / "webview"
    (root / "EBWebView").mkdir(parents=True)  # layout of earlier versions
    (root / "4000000001").mkdir()  # an instance that is gone
    (root / "4000000002").mkdir()  # an instance still open
    (root / "4000000002" / "Local State").write_text("{}")
    monkeypatch.setattr(server, "pid_alive", lambda pid: pid == 4000000002)
    folder = app.webview_storage()
    assert folder == root / str(os.getpid())
    assert sorted(p.name for p in root.iterdir()) == sorted([str(os.getpid()), "4000000002"])
    assert (root / "4000000002" / "Local State").is_file()


def test_webview2_starts_with_its_microsoft_connections_off():
    # D11: what turns WebView2's own traffic off. Losing one of these flags brings a connection back.
    args = app.WEBVIEW2_ARGS
    features = args.split("--disable-features=", 1)[1].split()[0].split(",")
    for feature in ("msSmartScreenProtection", "msOneAuthWAM", "msLoadOneAuthInBackground", "msImplicitSignin"):
        assert feature in features
    assert "ElasticOverscroll" in features  # pywebview's own flag, which the variable replaces
    for flag in ("--disable-background-networking", "--disable-component-update", "--no-pings", "--disable-breakpad"):
        assert flag in args.split()
    assert "--no-proxy-server" in args and '--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"' in args


class FakeWindow:
    def __init__(self, answer: bool):
        self.answer, self.asked = answer, 0

    def create_confirmation_dialog(self, title, message):
        self.asked += 1
        return self.answer


def test_browser_run_stops_cleanly_when_the_url_file_is_deleted(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(server, "setup_logging", lambda: None)  # keep the developer's log untouched
    monkeypatch.setattr(server, "cleanup_stale_sessions", lambda: 0)
    url_file = tmp_path / "url.json"
    sessions = lambda: set(Path(tempfile.gettempdir()).glob(server.SESSION_PREFIX + "*"))  # noqa: E731
    before = sessions()
    result = []
    argv = ["--browser", "--no-open", "--engine", "fake", "--url-file", str(url_file)]
    runner = threading.Thread(target=lambda: result.append(app.main(argv)), daemon=True)
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while not url_file.exists() and runner.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert url_file.exists()
        mine = sessions() - before
        assert len(mine) == 1
        url_file.unlink()
        runner.join(timeout=20)
        assert not runner.is_alive() and result == [0]
        assert not any(folder.exists() for folder in mine)  # the working copies went with the session
    finally:
        url_file.unlink(missing_ok=True)
        runner.join(timeout=20)


def test_closing_asks_only_when_work_would_be_lost():
    session = server.Session(FakeEngine())
    try:
        window = FakeWindow(answer=False)
        assert app.confirm_close(window, session) is True  # nothing loaded
        file = session.new_file("informe.pdf")
        session.files[file.id] = file  # "queued": added but never processed
        assert app.confirm_close(window, session) is True and window.asked == 0
        for status in ("processing", "ready", "confirmed"):
            file.status = status
            assert app.confirm_close(window, session) is False
        file.status = "exported"
        assert app.confirm_close(window, session) is True
        assert window.asked == 3
    finally:
        session.close()


@pytest.fixture
def client():
    with LiveClient(create_app(FakeEngine(), TOKEN)) as c:
        c.headers["X-Session-Token"] = TOKEN
        yield c


def test_export_to_a_folder_that_denies_writing_fails_at_once(client, tmp_path, monkeypatch):
    file_id = upload(client, "informe.pdf", make_pdf())
    client.post("/api/process", json={"file_ids": [file_id]})
    assert wait_status(client, file_id)["status"] == "ready"
    client.post(f"/api/files/{file_id}/confirm")
    real_open = os.open
    attempts = []

    def denied(path, *args, **kwargs):
        if os.path.basename(path).startswith(".anonimizador_"):
            attempts.append(path)
            raise PermissionError(13, "Access is denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied)
    started = time.monotonic()
    r = client.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": False, "audit_json": False})
    assert r.status_code == 422 and r.json()["error"] == "dest_not_writable"
    assert time.monotonic() - started < 5 and len(attempts) == 1


def test_invalid_requests_do_not_log_what_was_sent(client, caplog):
    file_id = upload(client, "informe.pdf", make_pdf())
    secret = "Comentario sobre Ana Prueba Inventada 12.345.678-5 " * 50  # longer than the 2000 allowed
    with caplog.at_level(logging.INFO, logger="anonymizer"):
        r = client.post(f"/api/files/{file_id}/findings", json={"page": 0, "polygon": [], "note": secret})
        assert r.status_code == 422
        r = client.put("/api/names", json={"entries": ["Ana Prueba Inventada"] * 20001})
        assert r.status_code == 422
    assert "invalid request" in caplog.text
    assert "Ana Prueba" not in caplog.text and "12.345.678-5" not in caplog.text
