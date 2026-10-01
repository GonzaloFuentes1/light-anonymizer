"""End to end: the local API with the real engine over the fictitious test set.

Uploads a scanned PDF, a photo shown rotated by its EXIF orientation, an identity-card photo and
a text PDF; processes them; reviews (removes one finding with a reason, draws one zone);
confirms; exports with the audit report; and checks the outputs. Every value comes from
``test_data/generated`` (invented data, see ``scripts/generate_test_data.py``) or from the API.

Slow (OCR and faces on real models): about a minute. Skipped when the models or the test set
are missing.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from anonymizer.api.server import create_app
from anonymizer.engine import real
from anonymizer.engine.model import DetectionOptions
from tests.live_client import LiveClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "test_data" / "generated"
MANIFEST = DATA / "manifest.json"
TOKEN = "token-de-prueba"

SCANNED = "pdf_scanned/informe_300dpi.pdf"
ROTATED = "exif/foto_orientacion_6.jpg"  # stored sideways, EXIF Orientation=6 shows it upright
ID_CARD = "id_card/cedula_frente_foto_perspectiva.jpg"
TEXT_PDF = "pdf_text/informe_honorarios_01.pdf"
FILES = (SCANNED, ROTATED, ID_CARD, TEXT_PDF)

REMOVAL_REASON = "No es un dato personal"
MANUAL_NOTE = "Encabezado del informe"

RUT_RE = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*[-‐‑–—]\s*[\dkK]\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(bool(real.missing_requirements()), reason="needs the OCR and face models"),
    pytest.mark.skipif(
        not MANIFEST.is_file(), reason="needs the test set: uv run python scripts/generate_test_data.py"
    ),
]


def manifest_entries() -> dict[str, dict]:
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    return {f["path"]: f for f in files if f["path"] in FILES}


def values(entry: dict, type_: str, level: str = "base") -> list[str]:
    return [e["value"] for e in entry["elements"] if e["type"] == type_ and e["level"] == level and e["value"]]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def ui_order(findings: list[dict]) -> list[dict]:
    """The order of the review list in the UI (``orderedVisible`` in app.js)."""

    def key(f):
        xs, ys = [p[0] for p in f["polygon"]], [p[1] for p in f["polygon"]]
        return f["page"], round(min(ys)), round(min(xs))

    return sorted((f for f in findings if f["doubtful"]), key=key) + sorted(
        (f for f in findings if not f["doubtful"]), key=key
    )


def wait_ready(client, ids: list[str], timeout: float = 900) -> dict[str, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = {f["id"]: f for f in client.get("/api/state").json()["files"] if f["id"] in ids}
        if all(f["status"] in ("ready", "error", "cancelled") for f in files.values()):
            return files
        time.sleep(0.25)
    raise AssertionError("the analysis did not finish in time")


@pytest.fixture(scope="module")
def session():
    """One app with the real engine and the four files analyzed (the slow part, done once)."""
    app = create_app(real.RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        assert client.get("/api/state").json()["engine"] == "real"
        names = [n.strip() for n in (DATA / "name_list.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
        assert client.put("/api/names", json={"entries": names}).status_code == 200
        ids = {}
        for rel in FILES:
            mime = "application/pdf" if rel.endswith(".pdf") else "image/jpeg"
            r = client.post("/api/files", files=[("files", (Path(rel).name, (DATA / rel).read_bytes(), mime))])
            assert r.status_code == 200, r.text
            ids[rel] = r.json()["files"][0]["id"]
        assert sorted(client.post("/api/process", json={}).json()["started"]) == sorted(ids.values())
        summaries = wait_ready(client, list(ids.values()))
        files = {rel: client.get(f"/api/files/{ids[rel]}").json() for rel in FILES}
        yield {"client": client, "ids": ids, "summaries": summaries, "files": files, "manifest": manifest_entries()}


def test_analysis(session):
    manifest = session["manifest"]
    for rel in FILES:
        file = session["files"][rel]
        assert file["status"] == "ready", (rel, file["step"], file["error_message"])
        assert file["kind"] == ("pdf" if rel.endswith(".pdf") else "image")
        assert file["timings"]["analyze"] > 0
        assert file["options"] == DetectionOptions().to_dict()  # the app's defaults
        # Pages in view space, as the test set describes them (after EXIF for the photo).
        expected = [(p["width"], p["height"]) for p in manifest[rel]["pages"]]
        assert [(p["width"], p["height"]) for p in file["pages"]] == pytest.approx(expected, abs=0.5)
        for f in file["findings"]:
            page = file["pages"][f["page"]]
            assert len(f["polygon"]) >= 3
            assert all(0 <= x <= page["width"] and 0 <= y <= page["height"] for x, y in f["polygon"]), (rel, f)
            # D12: URLs that are not personal start suggested (shown, not applied).
            assert f["status"] == ("suggested" if f["optional"] else "proposed")
            assert f["history"][0]["action"] == f["status"]
            assert bool(f["doubt_reason"]) == f["doubtful"]
    # Every stage that ran was measured; the scan and the photos went through OCR, faces and QR.
    for rel in (SCANNED, ROTATED, ID_CARD):
        timings = session["files"][rel]["timings"]
        assert {"render", "ocr", "faces", "qr"} <= set(timings) and timings["ocr"] > 0
    assert set(session["files"][TEXT_PDF]["timings"]) >= {"text", "analyze"}

    types = {rel: {f["type"] for f in session["files"][rel]["findings"]} for rel in FILES}
    assert {"rut", "email", "phone"} <= types[SCANNED]
    assert {"rut", "email", "phone"} <= types[ROTATED]
    assert {"rut", "face"} <= types[ID_CARD]
    assert {"rut", "email", "phone", "name"} <= types[TEXT_PDF]
    assert len(set(f["page"] for f in session["files"][SCANNED]["findings"])) == 2

    # Text-layer findings carry the exact text of the test set.
    found = {f["text"] for f in session["files"][TEXT_PDF]["findings"]}
    for type_ in ("rut", "email"):
        assert set(values(manifest[TEXT_PDF], type_)) <= found

    # OCR read the same line in several orientations: the reviewer sees it once.
    for rel in (SCANNED, ROTATED, ID_CARD):
        ruts = [f for f in session["files"][rel]["findings"] if f["type"] == "rut"]
        assert len({(f["page"], squash(f["text"] or "")) for f in ruts}) == len(ruts), (rel, ruts)

    # The review list shows the doubtful findings first; there are some (an invalid check digit,
    # names found only by context).
    for rel in FILES:
        ordered = ui_order(session["files"][rel]["findings"])
        flags = [f["doubtful"] for f in ordered]
        assert flags == sorted(flags, reverse=True)
        assert session["summaries"][session["ids"][rel]]["counts"]["doubtful"] == sum(flags)
    assert any(f["doubtful"] for f in session["files"][TEXT_PDF]["findings"])


def test_render_pages(session):
    client, ids = session["client"], session["ids"]
    r = client.get(f"/api/files/{ids[SCANNED]}/pages/1.png?zoom=0.5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (298, 421)
    page = session["files"][ROTATED]["pages"][0]
    png = client.get(f"/api/files/{ids[ROTATED]}/pages/0.png").content
    assert Image.open(io.BytesIO(png)).size == (page["width"], page["height"])
    with Image.open(DATA / ROTATED) as raw:
        assert raw.size == (page["height"], page["width"])  # stored sideways


def test_review_and_export(session, tmp_path):
    client, ids, manifest = session["client"], session["ids"], session["manifest"]
    text_id = ids[TEXT_PDF]

    # The reviewer keeps a doubtful RUT visible (e.g. it is a code, not a person's RUT).
    file = client.get(f"/api/files/{text_id}").json()
    removed = next(f for f in ui_order(file["findings"]) if f["doubtful"] and f["type"] == "rut")
    r = client.patch(
        f"/api/files/{text_id}/findings/{removed['id']}",
        json={"action": "remove", "reason": REMOVAL_REASON, "note": "Revisado contra el original"},
    )
    assert r.status_code == 200 and r.json()["status"] == "removed"

    # And draws a zone over the first line of the page (not proposed by the engine).
    header = next(e for e in manifest[TEXT_PDF]["elements"] if e["type"] == "text" and e["page"] == 0)
    xs, ys = [p[0] for p in header["polygon"]], [p[1] for p in header["polygon"]]
    zone = [
        [min(xs) - 2, min(ys) - 2],
        [max(xs) + 2, min(ys) - 2],
        [max(xs) + 2, max(ys) + 2],
        [min(xs) - 2, max(ys) + 2],
    ]
    r = client.post(f"/api/files/{text_id}/findings", json={"page": 0, "polygon": zone, "note": MANUAL_NOTE})
    assert r.status_code == 200 and r.json()["type"] == "manual"
    manual_id = r.json()["id"]

    for rel in FILES:
        r = client.post(f"/api/files/{ids[rel]}/confirm")
        assert r.status_code == 200 and r.json()["status"] == "confirmed", (rel, r.text)

    dest = tmp_path / "publicar"
    r = client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": True, "audit_json": True})
    assert r.status_code == 200, r.text
    body = r.json()
    results = {res["file_id"]: res for res in body["results"]}
    for rel in FILES:
        res = results[ids[rel]]
        assert res["exported"] is True and res["leaks"] == [], (rel, res)
        assert Path(res["output_path"]).parent == dest.resolve()
        assert Path(res["output_path"]).name == Path(rel).name
    assert (DATA / TEXT_PDF).read_bytes() != Path(results[text_id]["output_path"]).read_bytes()

    # PDFs: no RUT or e-mail left as text (except the one kept by the reviewer), no metadata.
    for rel in (SCANNED, TEXT_PDF):
        with pymupdf.open(results[ids[rel]]["output_path"]) as doc:
            text = "\n".join(page.get_text() for page in doc)
            in_zone = doc[0].get_text(clip=pymupdf.Rect(zone[0], zone[2])) if rel == TEXT_PDF else ""
            assert not any(v for k, v in doc.metadata.items() if k not in ("format", "encryption")), rel
            assert not doc.get_xml_metadata(), rel
        flat = squash(text)
        for type_ in ("rut", "email"):
            for value in values(manifest[rel], type_):
                if squash(value) != squash(removed["text"]):
                    assert squash(value) not in flat, (rel, type_)
        leftover = [m for m in RUT_RE.findall(text) if squash(m) != squash(removed["text"])]
        assert leftover == [] and EMAIL_RE.findall(text) == [], rel
        if rel == TEXT_PDF:
            assert squash(removed["text"]) in flat  # kept visible on purpose
            assert in_zone.strip() == ""  # covered by the manual zone
            with pymupdf.open(DATA / rel) as original:
                assert original[0].get_text(clip=pymupdf.Rect(zone[0], zone[2])).strip()

    # Images: same view size, no EXIF or other metadata, and the face is painted over.
    for rel in (ROTATED, ID_CARD):
        file = session["files"][rel]
        with Image.open(results[ids[rel]]["output_path"]) as out:
            assert out.size == (file["pages"][0]["width"], file["pages"][0]["height"])
            assert len(out.getexif()) == 0
            assert not {"exif", "xmp", "icc_profile", "photoshop"} & set(out.info)
            rgb = out.convert("RGB")
        if rel == ID_CARD:
            face = next(f for f in file["findings"] if f["type"] == "face")
            fx, fy = [p[0] for p in face["polygon"]], [p[1] for p in face["polygon"]]
            center = (round((min(fx) + max(fx)) / 2), round((min(fy) + max(fy)) / 2))
            assert max(rgb.getpixel(center)) < 30

    # The audit report records the removal with its reason and the manual zone.
    report = json.loads(Path(body["audit"]["json_path"]).read_text(encoding="utf-8"))
    assert Path(body["audit"]["pdf_path"]).is_file()
    records = {rec["id"]: rec for rec in report["files"]}
    assert set(records) == set(ids.values())
    record = records[text_id]
    assert record["leak_check"]["passed"] is True
    changes = record["reviewer_changes"]
    assert any(
        c["finding_id"] == removed["id"] and c["action"] == "removed" and c["reason"] == REMOVAL_REASON for c in changes
    )
    assert any(c["finding_id"] == manual_id and c["action"] == "added" and c["note"] == MANUAL_NOTE for c in changes)
    manual = next(f for f in record["findings"] if f["id"] == manual_id)
    assert manual["type"] == "manual" and manual["applied"] is True
    assert record["removed_by_reviewer"] == 1 and record["added_by_reviewer"] == 1
    for rec in records.values():
        assert rec["exported"] is True and rec["leak_check"]["passed"] is True

    state = {f["id"]: f["status"] for f in client.get("/api/state").json()["files"]}
    assert set(state.values()) == {"exported"}
