# Licenses

**Project license: GNU AGPL-3.0-or-later** (see [LICENSE](LICENSE)). The application uses
PyMuPDF, which is AGPL-3.0, so the project as a whole is licensed under the AGPL (decision D1,
2026-10-01). The source code is on GitHub (https://github.com/GonzaloFuentes1/light-anonymizer),
which is what the AGPL requires when the application is given to another institution: every
executable says in its `LEEME.txt` which commit it was built from.

Dependency policy: every other component that ships (libraries and models) must be under a
license compatible with the AGPL and with public and government use (MIT, Apache 2.0, BSD or
similar). Nothing under a non-commercial license and nothing from InsightFace.

This file distinguishes three things: what **will ship** with the application, what is used
only **for development and testing**, and the **test data**. Licenses were verified on
2026-09-30 against the license files inside each package (not just the PyPI field, which is
sometimes wrong), and every critical conclusion was verified independently.

**The Windows executable** carries the license texts of everything it bundles in
`THIRD_PARTY_LICENSES/`, next to the `.exe`: `scripts/collect_licenses.py` copies the license,
notice and authors files of every bundled package (`dist-info/licenses`, onnxruntime's
`ThirdPartyNotices.txt`, OpenCV's `LICENSE-3RD-PARTY.txt`...) and the texts their wheels lack,
kept in `packaging/licenses/` (rapidocr, antlr4, proxy_tools, the WebView2 SDK, the libraries
compiled into MuPDF, the Python runtime's libraries, the models). Its `INDEX.txt` lists each
component with its version, license and source package, and is the authoritative inventory of
what ships; this file records the decisions. The build fails if a bundled package has no license
text, or if a document, image or test file ends up in the bundle.

## 1. Components planned for the application

Status: ✅ compatible · ⚠️ compatible with conditions · ⛔ not used.

| Component | Version | License | Status | Notes |
|---|---|---|---|---|
| **PyMuPDF** (MuPDF) | 1.28.2 | AGPL-3.0 or Artifex commercial license | ✅ | Used under the AGPL; for that reason the whole project is licensed under AGPL-3.0-or-later and its source is public (decision D1). |
| pypdfium2 (PDFium) | 5.13.0 (PDFium 153.0.7999.0) | BSD-3-Clause / Apache-2.0 | ✅ | Bundles, among others, freetype (used under the FTL option), ICU, lcms, libjpeg-turbo, openjpeg, libpng, libtiff and zlib, all permissive. |
| pypdf | 6.19.0 | BSD-3-Clause | ✅ | Candidate for cleaning the PDF structure (metadata, XMP, annotations, attachments, JavaScript, layers). |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (OpenCV), MIT (packaging) | ⚠️ Windows / ⛔ macOS | Windows: bundles FFmpeg (LGPL-2.1) in a single video DLL that can be removed (YuNet was verified to keep working), and links Intel IPP ICV under the *Intel Simplified Software License* (not OSI). The executable leaves that DLL out (the spec drops it and `build_exe.py` checks), but it carries IPP ICV 2026.0.0 inside `cv2.pyd`. **Open issue (not settled by D3, which covers weak copyleft only):** IPP is statically linked into `cv2.pyd` and its license forbids modification and reverse engineering, restrictions the AGPL does not allow in the same program as AGPL MuPDF; the fix is a build of OpenCV without IPP (`-DWITH_IPP=OFF`) or dropping OpenCV (YuNet on onnxruntime). macOS (not a version-1 target, decision D4): the PyPI wheels link a **GPL-3.0** FFmpeg with x264/x265 that cannot be removed. Alternative: build OpenCV without FFmpeg or IPP, or run YuNet directly with onnxruntime. |
| onnxruntime | 1.30.0 | MIT | ✅ | Includes Eigen (MPL-2.0, headers only), which is complied with by keeping the notice. Requires macOS 14 or later (Apple Silicon). **Ships Microsoft telemetry** (ETW on Windows; HTTPS upload on macOS since 1.29), which the app disables at startup (PLAN.md, 5.6). |
| rapidocr | 3.9.2 | Apache-2.0 | ⚠️ | The package itself is compatible, but it requires opencv-python with a GUI (Qt and FFmpeg), shapely (GEOS, LGPL-2.1), requests and certifi (MPL-2.0) and tqdm (MPL-2.0); the weak-copyleft ones are accepted (D3) and ship with their licenses. It also includes code that downloads models and opens URLs (the app passes the models by path, so none is downloaded). Proposal: use its models and port only the inference (PLAN.md, section 5.4). |
| PaddleOCR models (ONNX) | PP-OCR | Apache-2.0 | ✅ | Per-model details in section 2. |
| YuNet (opencv_zoo) | 2023mar / 2026may | MIT | ✅ | Copyright (c) 2020 Shiqi Yu. |
| numpy | 2.5.3 | BSD-3-Clause and other permissive licenses | ✅ | OpenBLAS (BSD-3); GCC runtime with the GCC exception. |
| Pillow | 12.3.0 | MIT-CMU | ✅ | Bundles freetype (FTL), harfbuzz, lcms2, libjpeg-turbo, libpng, libwebp, openjpeg, libtiff, zlib-ng, xz, all permissive. Does not read HEIC. |
| pillow-heif | 1.8.0 | BSD-3 (source) / **GPL-2.0 (wheels, because of x265)** | ⛔ | Will not be used. |
| pi-heif | 1.4.0 | BSD-3 (source) / LGPL-3.0 (libheif, libde265) | ⚠️ | Decode only. Acceptable only if LGPL is approved, with folder-mode packaging (not a single file). **Decision pending** (D5). |
| rapidfuzz | 3.14.6 | MIT | ✅ | Fuzzy name matching. |
| pyclipper | 1.4.0 | MIT | ✅ | Polygon expansion for the text detector (replaces shapely). |
| FastAPI | 0.142.2 | MIT | ✅ | Now requires `opentelemetry-api` (Apache-2.0), which sends nothing without the SDK. The app turns FastAPI's telemetry off anyway (`telemetry=` all off, `OTEL_SDK_DISABLED`), so no exporter can be set up from the environment. |
| Starlette / Uvicorn / Pydantic / python-multipart | 1.7.0 / 0.54.0 / 2.13.5 / 0.0.32 | BSD-3 / BSD-3 / MIT / Apache-2.0 | ✅ | |
| pywebview | 6.2.1 | BSD-3-Clause | ✅ | Includes Microsoft's WebView2 SDK (BSD-style license; the notice is reproduced). On Windows it uses pythonnet (MIT), clr-loader (MIT), proxy-tools (MIT) and bottle (MIT). On macOS it uses pyobjc (MIT). |
| PyInstaller | 6.22.3 | GPL-2.0 with a bootloader exception | ✅ | It is a build tool; the exception covers what ends up inside the executable. |
| reportlab | 5.0.1 | BSD | ⚠️ | Includes the DarkGarden font (GPL-2.0), which will be excluded from the package if reportlab is used for the PDF report. |
| piexif | 1.1.3 | MIT | ✅ | |
| qrcode / segno | 8.2 / 1.6.6 | BSD-3 / BSD-3 | ✅ | Only if QR generation is needed. QR reading is done by OpenCV. |
| python-phonenumbers | 9.0.40 | Apache-2.0 | ✅ | Optional, as support for the phone detector. |

Not in the current executable: pypdfium2 and piexif (test bench only), pypdf, rapidfuzz,
reportlab, pillow-heif, pi-heif, qrcode, segno and python-phonenumbers.

Weak-copyleft components (LGPL, MPL) are accepted when their license is complied with and they
stay replaceable (decision D3, 2026-10-05). rapidocr pulls in shapely/GEOS (LGPL-2.1, shipped as
separate DLLs in `Shapely.libs/` that can be replaced in the one-folder layout), certifi and tqdm
(MPL-2.0), plus requests; onnxruntime compiles in Eigen (MPL-2.0, headers only). They ship in the
executable with their license texts, and `THIRD_PARTY_LICENSES/INDEX.txt` says where their source
code is. OpenCV's FFmpeg DLL (LGPL-2.1) is not bundled: the spec drops it and the build fails if
it shows up. img2pdf (LGPL-3.0) and pyzbar/zbar (LGPL-2.1) are not used.

Intel IPP is not weak copyleft and is **still open** (D3 does not cover it): the executable
bundles `cv2.pyd` of opencv-python-headless 5.0.0.93, with IPP ICV 2026.0.0 linked statically
(`cv2.getBuildInformation()`); its terms ship in OpenCV's `LICENSE-3RD-PARTY.txt`.

## 2. Models

| Model | Origin | License | Use |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` (SHA-256 `8f2383e4…2fa4`) | opencv_zoo, `models/face_detection_yunet` | MIT | Face detection (fixed-size input). |
| `face_detection_yunet_2026may.onnx` (SHA-256 `ebafce4e…f0f0`, 229 738 bytes) | opencv_zoo | MIT | Face detection (dynamic input, recommended for OpenCV 5). |
| `PP-OCRv6_det_small.onnx` (9 929 594 bytes) and `ch_PP-OCRv5_det_mobile.onnx` (4 819 576 bytes) | PaddleOCR, converted to ONNX by RapidAI (included in `rapidocr` 3.9.2 / ModelScope) | Apache-2.0 | Text detector; one will be chosen in phase 1. |
| `PP-OCRv6_rec_small.onnx` (21 234 383 bytes, embedded dictionary of 18 708 characters) | same | Apache-2.0 | Multilingual recognizer with accents and ñ/Ñ. |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` (585 532 bytes) | same | Apache-2.0 | 0/180 orientation classifier. |

The models are not committed to the repository: `scripts/download_models.py` (phase 1)
downloads them for development and verifies their SHA-256; packaging bundles them inside the
executable.

## 3. Development and testing tools (not shipped)

| Tool | License | Use |
|---|---|---|
| PyMuPDF | AGPL-3.0 | Besides being the application's PDF engine (section 1), the test bench uses it to generate test PDFs (layers, annotations, incremental revisions) and as one of the evaluator's two independent extractors. |
| pypdfium2 | BSD-3 / Apache-2.0 | The evaluator's second independent extractor. |
| matplotlib | Matplotlib License (PSF) | Only for its DejaVu and STIX fonts, used to draw the test data. |
| piexif, OpenCV, numpy, Pillow | see above | Test-data generation. |
| pytest, ruff | MIT | Tests and style. |

## 4. Test data (not shipped)

All personal data in the test set is invented. Face images are downloaded to
`test_data/cache/faces/` (outside version control), verified by SHA-256 and **are not included
in the application or the installer**. The full catalog, with the URL, SHA-256 and attribution
of every image, is in `test_bench/face_sources.json`.

| Source | Images | License | Attribution |
|---|---|---|---|
| Face Research Lab London Set | 16 (8 frontal, 5 three-quarter, 3 profile) | CC BY 4.0 | DeBruine, L. M. & Jones, B. C. (2017). *Face Research Lab London Set*. figshare. doi:10.6084/m9.figshare.5047666.v5. The people signed consent for use "in lab-based and web-based studies in their original or altered forms and to illustrate research". These are the only faces pasted into fictitious documents (ID card, record sheets). |
| Open Images V7 (validation) | 5 group photos (70 hand-annotated faces) | Images CC BY 2.0 (Flickr authors, see catalog); boxes CC BY 4.0 (Google LLC) | Each image carries its author, title and original URL in the catalog. They are used as-is, never pasted into documents. |
| Official US portraits (Congress, Senate) and NASA | 4 (including a group photo of 43 people) | Public domain (work of the US government) | United States Congress / United States Senate / NASA, via Wikimedia Commons. Public domain covers copyright, not personality rights: they are used only as internal test photos, never in fictitious documents. |
| DejaVu and STIX fonts (in matplotlib) | — | Bitstream Vera / public domain (DejaVu); SIL OFL 1.1 (STIX) | Typefaces used to draw the fictitious documents. |

Rejected because of their license: DigiFace-1M and FaceSynthetics (Microsoft, non-commercial
research only), FFHQ (CC BY-NC-SA) and any data derived from InsightFace. SFHQ (synthetic
faces, labeled CC0) is excluded by default because it was generated with StyleGAN2, whose
code has a non-commercial license.
