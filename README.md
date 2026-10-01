# Light Anonymizer

[Español](README.es.md) · **English**

A desktop tool that anonymizes PDFs and images **locally**, built for Chilean public officials who publish documents under transparency rules (CoP 33 / SmartGORE). It finds personal data — RUT, email, phone, URL, names from a list, faces, and text inside scans and photos — removes it for real (not with black boxes drawn on top), strips metadata, and then re-checks the output for leaks. Nothing leaves the computer.

> **Status: phase 0 finished, phases 1–2 in progress, phase 3 pending.** The repository contains the test bench (a fictitious test set with exact ground truth, the evaluator and baselines), the engine and a preliminary desktop app that runs it. There is no installer yet. Human review of every document before publishing is mandatory, always.

## Principles

1. **Recall over precision.** When in doubt, redact. Validations (RUT check digit, OCR confidence, face score) only order the review; they never discard a finding.
2. **Real redaction.** Content under a redacted area is deleted from the file (text, image pixels, vector paths); the file is fully rewritten so no previous version survives.
3. **Metadata scrubbing.** EXIF (including GPS and thumbnails), XMP, PDF metadata, annotations, attachments, hidden layers, forms, bookmarks and JavaScript are removed.
4. **Automatic leak check.** The output is scanned again with the same detectors plus byte-level and structural checks; anything found is reported as a leak.
5. **Offline.** No network calls at run time, models are bundled, no telemetry.
6. **Free software.** The project is AGPL-3.0; every bundled component has an AGPL-compatible license and nothing is non-commercial (see [LICENSES.md](LICENSES.md)).
7. **Human review is mandatory.** The app never says "clean document"; it says "review ready to confirm".

## Repository layout

```
anonymizer/      engine (engine/real.py: detect + apply + leak check), local API and preliminary app
test_bench/      development tooling: test-set generators, evaluator, baselines, prototype
  generators/    one module per document family (text PDFs, scans, rotated images, EXIF, ID card, screenshots, faces…)
  evaluation/    recall and leak checks (text, bytes, pixels, covered images, orphan images, vector paths, metadata)
  baselines/     identity, oracle, notebook and prototype (runs the real engine)
scripts/         generate_test_data.py, download_models.py, build_exe.py, demo_notebook_leak.py
packaging/       PyInstaller spec and launcher of the Windows executable, and the license texts its wheels lack
tests/           pytest suite
docs/            metric definition
test_data/       generated test set and caches (not versioned)
models/          ONNX models (not versioned; downloaded by script)
results/         outputs of local runs (not versioned)
```

## Quick start (development)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --all-groups                        # dependencies
uv run python scripts/download_models.py    # YuNet face model, SHA-256 checked
uv run python scripts/generate_test_data.py # fictitious test set -> test_data/generated
uv run python -m anonymizer.app             # desktop app (--browser to open it in the browser instead)

# run a system over the test set and evaluate it
uv run python -m test_bench.baseline prototype --manifest test_data/generated/manifest.json --output results/details/prototype --processes 3
uv run python -m test_bench.evaluate --manifest test_data/generated/manifest.json \
    --redaction-report results/details/prototype/report.json \
    --outputs-dir results/details/prototype/files --report-dir results/details/prototype

uv run python -m test_bench.compare         # before/after sheets -> results/examples
uv run python -m test_bench.process_folder <folder with your documents> --output results/gore
uv run pytest -q                            # tests
```

Use at most 3 parallel processes on a machine with 8 GB of RAM (each OCR worker uses ~600 MB).

## Building the Windows executable

A preliminary Windows build (there is no installer yet), made with PyInstaller in one-folder mode:

```bash
uv run python scripts/download_models.py           # once: YuNet face model, SHA-256 checked
uv run --group build python scripts/build_exe.py   # about 2 minutes, from a clean working copy
```

The AGPL requires offering the exact source of what is handed out, so the script refuses a working copy with uncommitted changes; `--allow-dirty` makes a test build, which its `LEEME.txt` marks as not for distribution. It checks the models (YuNet's SHA-256 and the PP-OCR models inside `rapidocr`), runs PyInstaller with `packaging/light_anonymizer.spec`, collects the third-party licenses (`scripts/collect_licenses.py`: the license files of every bundled package, plus the texts in `packaging/licenses/` that their wheels lack; the build fails if a bundled package has none), checks that nothing development-only or document-like was bundled (test bench, test data, pytest, matplotlib, OpenCV's FFmpeg DLL, any PDF, image or office file, a dotted RUT in the app's code…) and writes:

- `dist/LightAnonymizer/LightAnonymizer.exe` plus its `_internal/` folder (about 260 MB): the app, without a console window. Next to it: `LEEME.txt` (how to open it, the license, no warranty, and the commit and URL of its source code, in Spanish), `LICENSE.txt`, `LICENSES.md` and `THIRD_PARTY_LICENSES/` (with `INDEX.txt`: each component, its version, license and source). Always copy the whole folder, not just the `.exe`.
- `dist/LightAnonymizer-<version>-windows.zip` (about 120 MB): the same folder, zipped, to hand out.

One folder rather than a single file: the app starts in one or two seconds instead of unpacking hundreds of MB to `%TEMP%` on every launch, and antivirus programs flag it less often (the first launch after unzipping is slower while the antivirus scans it). It needs the WebView2 runtime, which Windows 10 and 11 include, and makes no network calls (WebView2 runs with its background services and its Windows-account sign-in turned off: with the sign-in on, it connected to Microsoft 365 on every launch). Unzip it to a short path outside OneDrive, such as `C:\Apps\LightAnonymizer`: with long paths Windows can fail to load native libraries.

The executable takes the same options as `python -m anonymizer.app` (`--browser`, `--no-open`, `--engine`), except that it always uses the real engine: it refuses `--engine fake`, ignores `ANONYMIZER_ENGINE`, and if the engine's components are missing (an antivirus may quarantine one) it shows an error instead of falling back to the development engine, which would export scanned pages and photos unredacted. Startup errors appear in a Spanish message box, and closing the window while files are being processed or are reviewed but not exported asks for confirmation. Because it has no console, `--url-file PATH` writes the URL and the session token to a JSON file for automated tests; nothing is written unless the flag is given, and the file is deleted when the app closes.

## The test bench

- **102 fictitious files** in 10 families with **971 personal-data elements** whose exact location is known, **44 hidden sensitive metadata items** and 7 files that must be rejected (password-protected, corrupt, empty, wrong format).
- Faces come from sources with documented free licenses (Face Research Lab London Set, Open Images, US public-domain portraits) and are downloaded and SHA-256 checked; they are never shipped.
- The evaluator measures **recall** (did the system find it?) and **leaks** (can it still be recovered from the output file?) — see [docs/metrics.md](docs/metrics.md). It validates itself: the *identity* baseline must leak everything and the *oracle* must leak nothing.
- Acceptance criterion for phase 1: zero leaks of RUT, email and phone at the base level, and zero metadata leaks.

## Privacy

Never commit real documents or anything derived from them (names, amounts, phrases). Real documents live only in `test_data/real/`, which is git-ignored, and so are all results. Tests and examples use invented data only.

## Documentation

- [PLAN.md](PLAN.md) — detailed plan, findings, and the open decisions for phases 1–3.
- [docs/metrics.md](docs/metrics.md) — how recall and leaks are measured.
- [LICENSES.md](LICENSES.md) — license of every dependency, model and test-data source.
- The end-user manual (in Spanish) will be written in phase 3.

## License

GNU Affero General Public License v3.0 or later (see [LICENSE](LICENSE)). Anyone who receives the application has the right to its source code, which is this repository. Third-party components keep their own licenses (see [LICENSES.md](LICENSES.md)).
