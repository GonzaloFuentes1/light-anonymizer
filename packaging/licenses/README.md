# License texts the installed packages do not ship

`scripts/collect_licenses.py` builds `dist/LightAnonymizer/THIRD_PARTY_LICENSES/` from the license
files inside every bundled package. The texts below are the ones those packages do not carry; they
were copied verbatim on 2026-10-01 from the sources listed here, so the build needs no network.

`packages/<distribution>/` is added to that distribution's folder; `components/<name>/` covers
what is not a Python package.

| Folder | What | Source |
|---|---|---|
| `packages/antlr4-python3-runtime/` | BSD-3-Clause (the wheel has no license file) | `LICENSE.txt` of antlr/antlr4 at tag 4.9.3 |
| `packages/proxy-tools/` | MIT (declared) and Werkzeug's BSD license (the code's origin); the package publishes no license file | Werkzeug 0.9.4 `LICENSE`; see the file's header |
| `packages/rapidocr/` | Apache-2.0 (the wheel has no license file) | `LICENSE` of RapidAI/RapidOCR at tag v3.9.2 |
| `packages/certifi/`, `packages/tqdm/` | Full MPL-2.0 text (their own files only point to it) | https://www.mozilla.org/media/MPL/2.0/index.txt |
| `packages/pywebview/webview2-sdk/` | Microsoft WebView2 SDK, whose DLLs pywebview bundles (`Microsoft.Web.WebView2.Core.dll`, `...WinForms.dll`, `WebView2Loader.dll`) | `LICENSE.txt` and `NOTICE.txt` of the NuGet package Microsoft.Web.WebView2 1.0.3856.49 (the version of the bundled DLLs) |
| `packages/pymupdf/mupdf-thirdparty/` | Libraries compiled into MuPDF (`mupdfcpp64.dll`) | MuPDF 1.28.2 `thirdparty/` submodules; repositories and commits in `SOURCES.txt` |
| `components/python/` | Software incorporated in the Python 3.12 runtime (OpenSSL 3, expat, libffi, zlib, libmpdec, ...), HACL* and XZ Utils' liblzma | CPython `Doc/license.rst` at tag v3.12.13; the MIT notice of `Modules/_hacl/Hacl_Hash_SHA2.c` (v3.12.13); `COPYING.0BSD` of tukaani-project/xz v5.8.1 |
| `components/models/` | YuNet (MIT, Shiqi Yu) and the PP-OCR models (Apache-2.0, PaddlePaddle Authors) | `models/face_detection_yunet/LICENSE` of opencv/opencv_zoo; `LICENSE` of PaddlePaddle/PaddleOCR |

The runtime's own `LICENSE.txt` (PSF license, bzip2, Tcl/Tk and Microsoft's distributable-code
terms for the MSVC runtime) is copied from `sys.base_prefix` at build time.

When a dependency is upgraded, check that its texts are still right (the WebView2 SDK version
comes with pywebview, the MuPDF libraries with PyMuPDF).
