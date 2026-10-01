# PyInstaller spec of the Windows executable: one folder (LightAnonymizer.exe plus _internal/), windowed.
#
#     uv run --group build python scripts/build_exe.py      # checks the models, builds, zips
#
# One folder rather than one file: the app starts without unpacking hundreds of MB to %TEMP% on
# every launch, antivirus programs flag it less often, and the bundled libraries stay replaceable.
# Resources keep their relative layout under _internal/ (sys._MEIPASS): see anonymizer/paths.py.
# PyInstaller runs this file with Analysis, PYZ, EXE, COLLECT, SPECPATH and DISTPATH predefined.

import importlib.util
import re
import shutil
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.win32 import versioninfo as vi

ROOT = Path(SPECPATH).resolve().parent
NAME = "LightAnonymizer"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
_numbers = [int(n) for n in re.match(r"\d+(?:\.\d+)*", VERSION).group().split(".")][:4]
VERSION_TUPLE = tuple(_numbers + [0] * (4 - len(_numbers)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Exactly the models the app uses (scripts/download_models.py is the single list; build_exe.py checks
# their SHA-256 before building).
YUNET_MODELS = [ROOT / "models" / name for name in _load_script("download_models").MODELS]

datas = [
    (str(ROOT / "anonymizer" / "ui"), "anonymizer/ui"),  # HTML, JS, CSS, fonts and their OFL texts
    *[(str(path), "models") for path in YUNET_MODELS],
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "LICENSES.md"), "."),
    # RapidOCR: its PP-OCR models (anonymizer/engine/ocr.py passes them by path, so nothing is
    # downloaded) and the YAML configuration it reads at start-up.
    *collect_data_files("rapidocr", includes=["config.yaml", "default_models.yaml", "models/*.onnx"]),
]

hiddenimports = [
    # Imported lazily by the app.
    "anonymizer.engine.real",
    "anonymizer.engine.fake",
    # uvicorn picks its event loop, protocols and lifespan by name at run time.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # pywebview's Windows backend: WinForms + Edge WebView2 through pythonnet / clr_loader (.NET Framework).
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "webview.platforms.win32",
    "clr",
    "pythonnet",
    "clr_loader",
    "clr_loader.netfx",
    # Engine libraries.
    "onnxruntime",
    "cv2",
    "pymupdf",
    "rapidocr.inference_engine.onnxruntime",
]

excludes = [
    # Development and test-data only.
    "test_bench",
    "tests",
    "scripts",
    "matplotlib",
    "pypdfium2",
    "piexif",
    "fontTools",  # only PyMuPDF's Document.subset_fonts (unused); comes with matplotlib
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "pygments",
    "ruff",
    "PyInstaller",
    "numpy.f2py",  # Fortran wrapper generator (a build tool)
    "IPython",
    "jupyter_client",
    "ipykernel",
    "tkinter",
    "_tkinter",
    "pydoc_data",
    # Build tools: only cffi's compiler support (unused) imports them.
    "setuptools",
    "pkg_resources",
    # pywebview back-ends for other platforms.
    "webview.platforms.android",
    "webview.platforms.cef",
    "webview.platforms.cocoa",
    "webview.platforms.gtk",
    "webview.platforms.qt",
    # RapidOCR inference engines other than onnxruntime.
    "rapidocr.inference_engine.mnn",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.pytorch",
    "rapidocr.inference_engine.tensorrt",
]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Files the app never uses, by their path inside the bundle:
# - OpenCV's video I/O DLL bundles FFmpeg (LGPL) and the app never reads video (YuNet works without it);
# - Pillow's AVIF codec (7.9 MB; AVIF is not an accepted input) and its Tk bridge;
# - pywebview's Android bridge, clr_loader's 32-bit DLL (the executable is 64-bit only) and pythonnet's
#   API documentation. pywebview's WebView2 loaders for every platform stay: it looks them all up at import.
UNUSED = (
    re.compile(r"(.*/)?opencv_videoio_ffmpeg[^/]*\.dll"),
    re.compile(r"PIL/_(avif|imagingtk)\.[^/]*pyd"),
    re.compile(r"webview/lib/pywebview-android\.jar"),
    re.compile(r"clr_loader/ffi/dlls/x86/.*"),
    re.compile(r"pythonnet/runtime/Python\.Runtime\.xml"),
)


def _used(entry):
    name = entry[0].replace("\\", "/")
    return not any(pattern.fullmatch(name) for pattern in UNUSED)


a.binaries = [entry for entry in a.binaries if _used(entry)]
a.datas = [entry for entry in a.datas if _used(entry)]

pyz = PYZ(a.pure)

version_info = vi.VSVersionInfo(
    ffi=vi.FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE),
    kids=[
        vi.StringFileInfo(
            [
                vi.StringTable(
                    "340A04B0",  # Spanish (Chile), Unicode
                    [
                        vi.StringStruct("CompanyName", "Gonzalo Fuentes"),
                        vi.StringStruct("FileDescription", "Anonimizador de documentos"),
                        vi.StringStruct("FileVersion", VERSION),
                        vi.StringStruct("InternalName", NAME),
                        vi.StringStruct("LegalCopyright", "© 2026 Gonzalo Fuentes. Licencia GNU AGPL-3.0-or-later."),
                        vi.StringStruct("OriginalFilename", f"{NAME}.exe"),
                        vi.StringStruct("ProductName", "Light Anonymizer"),
                        vi.StringStruct("ProductVersion", VERSION),
                        vi.StringStruct("Comments", "Software libre (GNU AGPL-3.0-or-later). Funciona sin internet."),
                    ],
                )
            ]
        ),
        vi.VarFileInfo([vi.VarStruct("Translation", [0x340A, 1200])]),
    ],
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed DLLs trigger antivirus false positives and some fail to load
    console=False,
    disable_windowed_traceback=True,  # packaging/launcher.py shows a Spanish message instead of a traceback
    icon="NONE",  # no icon yet: the generic Windows one rather than PyInstaller's
    version=version_info,
    contents_directory="_internal",
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)

# The license texts also go next to the .exe, where whoever receives the folder sees them.
_app_dir = Path(DISTPATH) / NAME
shutil.copyfile(ROOT / "LICENSE", _app_dir / "LICENSE.txt")
shutil.copyfile(ROOT / "LICENSES.md", _app_dir / "LICENSES.md")
