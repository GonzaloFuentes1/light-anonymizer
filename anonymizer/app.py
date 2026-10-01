"""Desktop entry point: local server on 127.0.0.1 plus a pywebview window.

    uv run python -m anonymizer.app              # desktop window
    uv run python -m anonymizer.app --browser    # development: open the UI in the default browser

``--engine fake`` forces the development engine (same as ``ANONYMIZER_ENGINE=fake``). Nothing
leaves the computer: the server only listens on 127.0.0.1, every API call needs a random
session token, and WebView2 runs with its background network features turned off.
"""

from __future__ import annotations

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"  # before anything can import onnxruntime

import argparse  # noqa: E402
import logging  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import webbrowser  # noqa: E402

from anonymizer.api import server  # noqa: E402

log = logging.getLogger("anonymizer")

WEBVIEW2_ARGS = (
    "--disable-features=ElasticOverscroll,msSmartScreenProtection --disable-background-networking "
    "--disable-component-update --no-pings --disable-domain-reliability --no-proxy-server"
)
FILE_TYPES = (
    "Documentos e imágenes (*.pdf;*.jpg;*.jpeg;*.png;*.webp;*.tif;*.tiff)",
    "Todos los archivos (*.*)",
)


def load_engine(choice: str | None):
    if choice:
        os.environ["ANONYMIZER_ENGINE"] = choice
    from anonymizer.engine import get_engine

    try:
        return get_engine()
    except ImportError:
        if os.environ.get("ANONYMIZER_ENGINE", "real") == "fake":
            raise
        log.warning("real engine not available yet, using the development engine")
        print("Aviso: el motor definitivo aún no está disponible; se usa el motor de prueba.", file=sys.stderr)
        from anonymizer.engine.fake import FakeEngine

        return FakeEngine()


def bind_socket() -> socket.socket:
    """A listening socket on a free port of 127.0.0.1 (no race between choosing and binding)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    return sock


class Server:
    def __init__(self, app, sock: socket.socket):
        import uvicorn

        self.sock = sock
        self.port = sock.getsockname()[1]
        config = uvicorn.Config(app, log_config=None, access_log=False, lifespan="on", timeout_graceful_shutdown=3)
        self.uvicorn = uvicorn.Server(config)
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self.uvicorn.run, kwargs={"sockets": [self.sock]}, daemon=True, name="server"
        )
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.uvicorn.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the local server did not start")
            time.sleep(0.05)

    def stop(self) -> None:
        self.uvicorn.should_exit = True
        if self.thread:
            self.thread.join(timeout=10)


class DesktopApi:
    """Methods the UI can call through ``window.pywebview.api`` (native dialogs)."""

    def __init__(self):
        self._window = None  # underscore: pywebview does not expose it to JavaScript

    def choose_files(self) -> list[str]:
        """Native open dialog (PDF and images, several at once). The UI sends the paths to
        ``POST /api/files/from-paths``."""
        import webview

        paths = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True, file_types=FILE_TYPES)
        return list(paths or [])

    def choose_folder(self) -> str | None:
        """Native folder dialog: a folder to add (from-paths) or the export destination."""
        import webview

        paths = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return paths[0] if paths else None


def wipe(folder) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anonymizer.app", description="Anonimizador")
    parser.add_argument("--browser", action="store_true", help="open the UI in the default browser (development)")
    parser.add_argument("--engine", choices=("fake", "real"), help="engine to use (default: ANONYMIZER_ENGINE)")
    parser.add_argument("--no-open", action="store_true", help="with --browser: only print the URL")
    args = parser.parse_args(argv)

    log_path = server.setup_logging()
    removed = server.cleanup_stale_sessions()
    log.info("starting (log %s, %d stale session folders removed)", log_path, removed)

    engine = load_engine(args.engine)
    token = secrets.token_urlsafe(32)
    launch_key = secrets.token_urlsafe(24)
    app = server.create_app(engine, token, launch_key=launch_key)
    session = app.state.session
    local = Server(app, bind_socket())
    url = f"http://127.0.0.1:{local.port}/"
    launch_url = f"{url}?k={launch_key}"

    try:
        local.start()
        log.info("server listening on 127.0.0.1:%d (engine %s)", local.port, session.engine_name)
        if args.browser:
            print(f"Anonimizador en {launch_url}", flush=True)
            print(f"Token de sesión: {token}", flush=True)
            print("Presiona Ctrl+C para cerrar.", flush=True)
            if not args.no_open:
                webbrowser.open(launch_url)
            try:
                while local.thread and local.thread.is_alive():
                    local.thread.join(timeout=0.5)
            except KeyboardInterrupt:
                pass
            return 0

        os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = WEBVIEW2_ARGS
        import webview

        storage = server.app_data_dir() / "webview"
        wipe(storage)
        api = DesktopApi()
        window = webview.create_window(
            "Anonimizador",
            launch_url,
            js_api=api,
            width=1360,
            height=860,
            min_size=(1100, 720),
            background_color="#F2F4F7",
            text_select=True,
        )
        api._window = window
        webview.start(private_mode=False, storage_path=str(storage))
        return 0
    finally:
        local.stop()
        session.close()
        log.info("stopped")


if __name__ == "__main__":
    sys.exit(main())
