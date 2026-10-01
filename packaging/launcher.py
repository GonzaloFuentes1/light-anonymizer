"""Entry script of the Windows executable (see light_anonymizer.spec): runs the desktop app.

The executable has no console, so a problem that stops the app is shown in a message box in
Spanish (the detail goes to the technical log). When the app has finished (window closed, server
stopped, working folder deleted, log flushed) the process ends at once: the interpreter's own
teardown (pythonnet, onnxruntime, about 1 GB of memory) kept it alive 4-14 s after the window
had gone.
"""

import logging
import multiprocessing
import os
import sys

TITLE = "Anonimizador"
UNEXPECTED = (
    "El Anonimizador se cerró por un problema inesperado. El detalle quedó en el registro técnico "
    "(%LOCALAPPDATA%\\Anonimizador\\logs).\n\n"
    "Si vuelve a ocurrir, descomprime de nuevo la carpeta completa de la aplicación."
)


def show_error(message: str) -> None:
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)  # MB_ICONERROR


def run() -> int:
    # Nothing starts child processes today; this keeps a future one from re-running the whole app.
    multiprocessing.freeze_support()
    try:
        from anonymizer.app import StartupError, main

        try:
            return main()
        except StartupError as exc:
            logging.getLogger("anonymizer").error("could not start: %s", exc)
            show_error(str(exc))
            return 1
    except Exception:
        logging.getLogger("anonymizer").exception("unexpected error")
        show_error(UNEXPECTED)
        return 1


if __name__ == "__main__":
    code = run()
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    os._exit(code)
