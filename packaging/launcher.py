"""Entry script of the Windows executable (see light_anonymizer.spec): runs the desktop app.

The executable has no console, so a problem that stops the app is shown in a message box in
Spanish (the detail goes to the technical log). When the app has finished (window closed, server
stopped, working folder deleted, log flushed) the process ends at once: the interpreter's own
teardown (pythonnet, onnxruntime, about 1 GB of memory) kept it alive 4-14 s after the window
had gone.

While it runs, every instance holds the named mutex ``APP_MUTEX`` (in its session and in the
global namespace): the installer and the uninstaller (``AppMutex`` in packaging/installer.iss)
see it and ask the user to close the app instead of replacing or deleting files in use.
"""

import logging
import multiprocessing
import os
import sys

TITLE = "Anonimizador"
# Must match AppMutex in packaging/installer.iss (tests/test_installer.py checks it).
APP_MUTEX = "LightAnonymizer.Running"
UNEXPECTED = (
    "El Anonimizador se cerró por un problema inesperado. El detalle quedó en el registro técnico "
    "(%LOCALAPPDATA%\\Anonimizador\\logs).\n\n"
    "Si vuelve a ocurrir, instala de nuevo la aplicación (o, si usas la versión .zip, descomprime "
    "de nuevo la carpeta completa)."
)


def show_error(message: str) -> None:
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)  # MB_ICONERROR


def hold_app_mutex() -> list[int]:
    """Creates (or opens) the app's named mutexes; Windows releases them when the process ends.

    Several instances may run at once: each one opens the same mutexes, which exist while any of
    them is alive. Returns the handles, never closed on purpose. A mutex that cannot be created is
    skipped: it only serves the installer.
    """
    import ctypes
    from ctypes import wintypes

    create = ctypes.WinDLL("kernel32").CreateMutexW  # own instance: the argtypes stay local
    create.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    create.restype = wintypes.HANDLE
    handles = []
    for name in (APP_MUTEX, "Global\\" + APP_MUTEX):
        handle = create(None, False, name)
        if handle:
            handles.append(handle)
    return handles


def run() -> int:
    # Nothing starts child processes today; this keeps a future one from re-running the whole app.
    multiprocessing.freeze_support()
    hold_app_mutex()  # first: the installer must see the app while it is still starting
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
