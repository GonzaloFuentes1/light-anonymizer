# Light Anonymizer

[Español](README.es.md) · **English**

A desktop tool that anonymizes PDFs and images **locally**, for Chilean public officials who publish documents under transparency rules. It finds personal data — RUT, email, phone, URL, names from a list, faces, handwritten and drawn signatures, and text inside scans and photos — removes it for real (not with black boxes drawn on top), strips metadata, and then re-checks the output for leaks. Nothing leaves the computer.

Human review of every document before publishing is mandatory: the app proposes, the reviewer decides.

**Platform:** Windows 10 and 11 (64-bit).

## How it works

1. **Choose files** — PDFs (with text or scanned), JPG, PNG, WEBP and TIFF, or whole folders.
2. **Process** — the app searches each file; you can start reviewing the ready ones while the rest finish.
3. **Review** — every page appears in one scroll, with the original and its marked zones on the left and the result exactly as it will be exported on the right. Doubtful findings come first; you can remove a censure (with a reason), add one by drawing a zone, or censor items that are not censored by default.
4. **Export** — the anonymized files go to a folder you choose, never over the originals, together with an audit report (PDF and JSON) of what was censored and what you changed.

## Principles

1. **Recall over precision.** When in doubt, redact. Validations (RUT check digit, OCR confidence, face score) only order the review; they never discard a finding.
2. **Real redaction.** Content under a redacted area is deleted from the file (text, image pixels, vector paths); the file is fully rewritten so no previous version survives.
3. **Metadata scrubbing.** EXIF (including GPS and thumbnails), XMP, PDF metadata, annotations, attachments, hidden layers, forms, bookmarks and JavaScript are removed.
4. **Automatic leak check.** The output is scanned again; a file with readable data left is not exported.
5. **Offline.** No network calls at run time, models are bundled, no telemetry.
6. **Free software.** AGPL-3.0; every bundled component has a compatible license (see [LICENSES.md](LICENSES.md)).
7. **Human review is mandatory.** The app never says "clean document"; it says "review ready to confirm".

**Some PDF pages may be exported as images.** When the app cannot be certain that nothing drawn is left under a black box (a letter drawn as a path that a box cuts, a stroke crossing the edge of a box, a pattern under it), that page is exported as a single image of the redacted page, with no text layer or vector content left. The review already shows it that way, and the audit report lists those pages and why.

**Doubtful RUTs are shown, not censored by default.** A number written as bare digits (no dots, no dash), whose check digit does not match and with no "RUT" label before it, is usually a folio or a code: it is listed with the items not censored by default, for the reviewer to decide. RUTs with dots or a dash and a wrong check digit are censored and marked doubtful.

## Installing

Run `LightAnonymizer-<version>-setup.exe`. It installs for the current user, without administrator rights, and adds "Anonimizador" and its user manual to the Start menu. The installer is not code-signed yet, so Windows may show "Windows protected your PC": click "More info", then "Run anyway". Uninstall from Settings > Apps > Installed apps.

The user manual, in Spanish: [docs/user-manual/manual-de-usuario.pdf](docs/user-manual/manual-de-usuario.pdf).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --all-groups                            # dependencies
uv run python scripts/download_models.py        # YuNet face model, SHA-256 checked
uv run python scripts/generate_test_data.py     # fictitious test set -> test_data/generated
uv run python scripts/generate_practice_docs.py # fictitious practice documents -> test_data/practice
uv run python -m anonymizer.app                 # desktop app (--browser to open it in the browser instead)
uv run pytest -q                                # tests

# run the engine over the test set and evaluate it
uv run python -m test_bench.baseline prototype --manifest test_data/generated/manifest.json --output results/details/prototype --processes 3
uv run python -m test_bench.evaluate --manifest test_data/generated/manifest.json \
    --redaction-report results/details/prototype/report.json \
    --outputs-dir results/details/prototype/files --report-dir results/details/prototype
```

Use at most 3 parallel processes on a machine with 8 GB of RAM (each OCR worker uses ~600 MB).

**The given-name dictionary** (`anonymizer/engine/data/given_names.txt`, versioned) holds the given names registered at least 100 times in Chile from 1920 to 2021, from the CC0 *guaguas* dataset (Servicio de Registro Civil e Identificación; see LICENSES.md). To rebuild it, run `uv run python scripts/build_given_names.py [--threshold 100]`: it downloads the dataset from a pinned commit, checks its SHA-256 and rewrites the file. Given names that are also ordinary words or places (Paz, Rosa, Santiago...) are listed in `anonymizer/engine/names.py` and do not start a name by themselves.

```
anonymizer/      engine, local API and desktop app
test_bench/      test-set generators, evaluator and baselines
scripts/         test data, models, user manual PDF, executable and installer builds
packaging/       PyInstaller spec, launcher, Inno Setup script and license texts
tests/           pytest suite
docs/            user manual
```

**The test bench** has 102 fictitious files in 10 families with 971 personal-data elements whose exact location is known, 44 hidden metadata items and 7 files that must be rejected. The evaluator measures **recall** (was it found?) and **leaks** (can it still be recovered from the output file?).

**Privacy.** Never commit real documents or anything derived from them. Real documents live only in `test_data/real/`, which is git-ignored, and so are all results. Tests and examples use invented data only.

## Building the Windows installer

```bash
uv run python scripts/download_models.py                       # once
uv run --group build python scripts/build_exe.py --installer   # from a clean working copy
```

It writes, in `dist/`, the app folder (`LightAnonymizer/`, with `LEEME.txt`, the licenses and the user manual), a zip of it, and `LightAnonymizer-<version>-setup.exe` (made with [Inno Setup 6](https://jrsoftware.org/isinfo.php)). The AGPL requires offering the exact source of what is handed out, so the build refuses a working copy with uncommitted changes; `--allow-dirty` makes a test build, marked as not for distribution. The build also checks the models and that nothing development-only or document-like is bundled.

Silent install: `LightAnonymizer-<version>-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`.

## Network traffic

The app makes no network calls: its local server listens only on 127.0.0.1, the models are bundled, the interface loads nothing from outside, and library telemetry is turned off. The window is Microsoft's WebView2, a Windows component; the app turns off its background services, sign-in and crash reporting, but WebView2 updates and Windows' own checks (SmartScreen, Defender) belong to the operating system. A machine that must have zero traffic needs an outbound firewall rule for `LightAnonymizer.exe`.

## License

GNU Affero General Public License v3.0 or later (see [LICENSE](LICENSE)). Anyone who receives the application has the right to its source code, which is this repository. Third-party components keep their own licenses (see [LICENSES.md](LICENSES.md)).
