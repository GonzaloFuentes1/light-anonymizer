"""Desktop entry point: local server on 127.0.0.1 plus a pywebview window.

    uv run python -m anonymizer.app              # desktop window
    uv run python -m anonymizer.app --browser    # development: open the UI in the default browser

``--engine fake`` forces the development engine (same as ``ANONYMIZER_ENGINE=fake``); the packaged
executable refuses it and never falls back to it (see :func:`load_engine`).
``--url-file PATH`` writes the launch URL and the session token to ``PATH`` (JSON) once the server
answers, for tests and automation: the windowed executable has no console to print them to, and
they are written nowhere unless this flag is given (the file is deleted when the app closes).
With ``--browser``, deleting that file stops the app cleanly (working folder deleted): the
windowed executable cannot receive Ctrl+C, and killing it leaves the working copies behind.

Nothing leaves the computer: the server only listens on 127.0.0.1, every API call needs a random
session token, and WebView2 runs with its background network and Windows-account features turned
off (``WEBVIEW2_ARGS``).
"""

from __future__ import annotations

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"  # before anything can import onnxruntime
os.environ["OTEL_SDK_DISABLED"] = "true"  # OpenTelemetry (comes with FastAPI): never set up an exporter

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

from anonymizer.api import server  # noqa: E402
from anonymizer.paths import is_frozen  # noqa: E402

log = logging.getLogger("anonymizer")

# WebView2 (Edge) flags: no background services (component updates, pings, SmartScreen) and no
# Windows-account features. With ``msOneAuthWAM`` on, the browser process signed in with the Windows
# work account by itself and opened a TLS connection to Microsoft 365 (52.97.x.x:443) a few seconds
# after every launch; that traffic goes through Windows, not Chromium's network stack, so only
# turning the feature off stops it. The host rules make any name lookup by Chromium fail (the UI is
# served from 127.0.0.1, which needs none). ``--disable-breakpad`` is Chromium's switch that turns
# crash reporting off: crash uploads use their own HTTP client, which the host rules do not cover
# (whether WebView2 honors it has not been measured; that needs a crash during a capture). What may
# still talk to Microsoft is listed in README.md, "Network traffic" (decision D11).
WEBVIEW2_ARGS = (
    "--disable-features=ElasticOverscroll,msSmartScreenProtection,msOneAuthWAM,msLoadOneAuthInBackground,"
    "msImplicitSignin,msEdgeOSAccountInfoSubstrate "
    "--disable-background-networking --disable-component-update --no-pings --disable-domain-reliability "
    "--disable-breakpad "
    '--no-proxy-server --host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"'
)
FILE_TYPES = (
    "Documentos e imágenes (*.pdf;*.jpg;*.jpeg;*.png;*.webp;*.tif;*.tiff)",
    "Todos los archivos (*.*)",
)


CLOSE_QUESTION = (
    "Hay archivos en proceso o revisados que todavía no se exportan. Si cierras ahora, ese trabajo "
    "se pierde.\n\n¿Quieres cerrar de todos modos?"
)
ENGINE_MISSING = (
    "Faltan componentes del motor de anonimización; puede que el antivirus haya bloqueado alguno.\n\n"
    "Instala de nuevo la aplicación (o, si usas la versión .zip, descomprime de nuevo la carpeta "
    "completa) y vuelve a abrirla."
)


class StartupError(Exception):
    """Keeps the app from starting; ``str(error)`` is the message for the user (Spanish)."""


def load_engine(choice: str | None):
    """The engine to use.

    In development the app falls back to the development engine when the real one cannot run. The
    packaged executable never does: that engine reads no scanned text and finds no faces, and its
    leak check only re-checks its own findings, so scanned pages and photos would be exported
    unredacted. There ``--engine fake`` is refused and ``ANONYMIZER_ENGINE`` is ignored.
    """
    if is_frozen():
        if choice == "fake":
            raise StartupError("Esta versión de la aplicación no incluye el motor de prueba.")
        os.environ["ANONYMIZER_ENGINE"] = "real"
    elif choice:
        os.environ["ANONYMIZER_ENGINE"] = choice
    from anonymizer.engine import get_engine

    engine = get_engine()
    if getattr(engine, "name", "") == "fake" and os.environ.get("ANONYMIZER_ENGINE", "real") != "fake":
        if is_frozen():
            raise StartupError(ENGINE_MISSING)
        print(
            "Aviso: faltan los modelos del motor definitivo; se usa el motor de prueba (sin OCR ni rostros).",
            file=sys.stderr,
        )
    return engine


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


def webview_storage() -> Path:
    """An empty WebView2 profile folder for this instance; those of instances that are gone are deleted.

    Each running instance has its own folder, and so its own WebView2 browser process: with a shared
    one, every launch wiped the profile under the instances already open. Falls back to the
    temporary folder when %LOCALAPPDATA% cannot be written.
    """
    for root in (server.app_data_dir() / "webview", Path(tempfile.gettempdir()) / "anonimizador_webview"):
        try:
            root.mkdir(parents=True, exist_ok=True)
            for child in root.iterdir():
                owner = int(child.name) if child.name.isdigit() else None
                if owner is not None and owner != os.getpid() and server.pid_alive(owner):
                    continue
                shutil.rmtree(child, ignore_errors=True)
            folder = root / str(os.getpid())
            folder.mkdir(exist_ok=True)
            return folder
        except OSError:
            log.warning("cannot use %s for the WebView2 profile", root, exc_info=True)
    raise StartupError("Windows no permite que la aplicación guarde sus archivos de trabajo.")


def unfinished_files(session: server.Session) -> int:
    """Files whose analysis or review is lost if the app closes now (processing, or not exported)."""
    with session.lock:
        return sum(
            1
            for file in session.files.values()
            if file.status in ("processing", "ready", "confirmed") or file.id in session.pending
        )


def confirm_close(window, session: server.Session) -> bool:
    """Asks before closing a window with unfinished work (pywebview cancels the close on False)."""
    if not unfinished_files(session):
        return True
    return bool(window.create_confirmation_dialog("Anonimizador", CLOSE_QUESTION))


def write_url_file(path: Path, url: str, token: str) -> None:
    """Writes ``{"url", "token", "pid"}`` to ``path`` in one step (a reader never sees half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps({"url": url, "token": token, "pid": os.getpid()}), encoding="utf-8")
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anonymizer.app", description="Anonimizador")
    parser.add_argument("--browser", action="store_true", help="open the UI in the default browser (development)")
    parser.add_argument("--engine", choices=("fake", "real"), help="engine to use (default: ANONYMIZER_ENGINE)")
    parser.add_argument("--no-open", action="store_true", help="with --browser: only print the URL")
    parser.add_argument(
        "--url-file", type=Path, help="write the launch URL and the session token to this file (JSON; for tests)"
    )
    args = parser.parse_args(argv)

    log_path = server.setup_logging()
    removed = server.cleanup_stale_sessions()
    log.info("starting (log %s, %d stale session folders removed)", log_path, removed)

    engine = load_engine(args.engine)
    token = secrets.token_urlsafe(32)
    launch_key = secrets.token_urlsafe(24)
    # Only the rates of the time estimate are saved there, one file per engine (never file names).
    estimates = server.app_data_dir() / f"estimates_{getattr(engine, 'name', 'engine')}.json"
    app = server.create_app(engine, token, launch_key=launch_key, estimates_path=estimates)
    session = app.state.session
    local = Server(app, bind_socket())
    url = f"http://127.0.0.1:{local.port}/"
    launch_url = f"{url}?k={launch_key}"

    try:
        local.start()
        log.info("server listening on 127.0.0.1:%d (engine %s)", local.port, session.engine_name)
        if args.url_file:
            write_url_file(args.url_file, launch_url, token)
        if args.browser:
            print(f"Anonimizador en {launch_url}", flush=True)
            print(f"Token de sesión: {token}", flush=True)
            print("Presiona Ctrl+C para cerrar.", flush=True)
            if not args.no_open:
                webbrowser.open(launch_url)
            try:
                while local.thread and local.thread.is_alive():
                    if args.url_file and not args.url_file.exists():
                        log.info("url file deleted: stopping")
                        break
                    local.thread.join(timeout=0.5)
            except KeyboardInterrupt:
                pass
            return 0

        os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = WEBVIEW2_ARGS
        import webview

        webview.settings["ALLOW_FILE_URLS"] = False  # no --allow-file-access-from-files: the UI never uses file://
        storage = webview_storage()
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

        def on_closing(window) -> bool:  # pywebview passes the window to a parameter named "window"
            return confirm_close(window, session)

        window.events.closing += on_closing
        webview.start(private_mode=False, storage_path=str(storage))
        shutil.rmtree(storage, ignore_errors=True)  # what WebView2 still holds goes at the next launch
        return 0
    finally:
        local.stop()
        session.close()
        if args.url_file:
            args.url_file.unlink(missing_ok=True)  # the token is useless now: do not leave it around
        log.info("stopped")


if __name__ == "__main__":
    sys.exit(main())
