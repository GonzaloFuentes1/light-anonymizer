"""End-to-end tests of the local API with the development engine (invented data only)."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from anonymizer.api.server import create_app, inject_token
from anonymizer.engine.fake import FakeEngine, rut_is_valid
from anonymizer.engine.model import AnalyzedFile, Finding
from tests.live_client import LiveClient

TOKEN = "token-de-prueba"
VALID_RUT = "12.345.678-5"
DOUBTFUL_RUT = "11.111.111-2"
EMAIL = "ana.prueba@ejemplo.cl"
PHONE = "+56 9 8123 4567"


def make_pdf(rotation: int = 0, password: str | None = None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Informe ficticio de prueba", fontsize=14)
    page.insert_text((72, 140), f"RUT: {VALID_RUT}", fontsize=11)
    page.insert_text((72, 160), f"Correo: {EMAIL}", fontsize=11)
    page.insert_text((72, 180), f"Teléfono: {PHONE}", fontsize=11)
    page.insert_text((72, 200), f"RUT contraparte: {DOUBTFUL_RUT}", fontsize=11)
    page.insert_text((72, 220), "Prestadora: Ana Prueba Inventada", fontsize=11)
    page.set_rotation(rotation)
    doc.set_metadata({"author": "Persona Inventada", "title": "Documento ficticio"})
    kwargs = {}
    if password:
        kwargs = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password, "owner_pw": password}
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


@pytest.fixture
def client():
    app = create_app(FakeEngine(), TOKEN)
    with LiveClient(app) as c:
        c.headers["X-Session-Token"] = TOKEN
        yield c


def upload(client, name: str, data: bytes, mime: str = "application/pdf") -> str:
    r = client.post("/api/files", files=[("files", (name, data, mime))])
    assert r.status_code == 200, r.text
    return r.json()["files"][0]["id"]


def wait_status(client, file_id: str, statuses=("ready", "error", "cancelled"), timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = next(f for f in client.get("/api/state").json()["files"] if f["id"] == file_id)
        if summary["status"] in statuses:
            return summary
        time.sleep(0.05)
    raise AssertionError(f"file {file_id} did not finish")


def test_rut_check_digit():
    assert rut_is_valid(VALID_RUT)
    assert not rut_is_valid(DOUBTFUL_RUT)
    assert rut_is_valid("11.111.111-1")


def test_token_required_and_headers():
    app = create_app(FakeEngine(), TOKEN)
    with LiveClient(app) as c:
        r = c.get("/api/state")
        assert r.status_code == 401
        assert r.json()["error"] == "unauthorized"
        assert "sesión" in r.json()["message"]
        assert c.get("/api/state", headers={"X-Session-Token": "otro"}).status_code == 401
        r = c.get("/api/state", headers={"X-Session-Token": TOKEN})
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-store"
        assert r.json() == {"files": [], "names_count": 0, "engine": "fake"}
        assert "access-control-allow-origin" not in r.headers
        # The index is public and carries the token.
        r = c.get("/")
        assert r.status_code == 200 and TOKEN in r.text
        assert r.headers["cache-control"] == "no-store"
        # Another host name (DNS rebinding) or another origin is refused.
        assert c.get("/", headers={"Host": "evil.example"}).status_code == 400
        r = c.post("/api/process", json={}, headers={"X-Session-Token": TOKEN, "Origin": "http://evil.example"})
        assert r.status_code == 403


def test_launch_key_protects_token():
    app = create_app(FakeEngine(), TOKEN, launch_key="llave")
    with LiveClient(app) as c:
        r = c.get("/")
        assert r.status_code == 403 and TOKEN not in r.text
        assert c.get("/?k=mala").status_code == 403
        r = c.get("/?k=llave")
        assert r.status_code == 200 and TOKEN in r.text  # redirect followed with the cookie


def test_inject_token():
    page = '<html><head><meta name="session-token" content=""></head></html>'
    assert 'content="abc"' in inject_token(page, "abc")
    assert 'content="abc"' in inject_token("<html><head></head></html>", "abc")


def test_full_flow(client, tmp_path: Path):
    r = client.put("/api/names", json={"entries": ["Ana Prueba Inventada", "  ", "ana prueba inventada"]})
    assert r.json() == {"entries": ["Ana Prueba Inventada"]}
    file_id = upload(client, "informe_ficticio.pdf", make_pdf())
    state = client.get("/api/state").json()
    assert state["names_count"] == 1
    assert state["files"][0]["status"] == "queued" and state["files"][0]["kind"] == "pdf"

    assert client.post("/api/process", json={}).json() == {"started": [file_id]}
    summary = wait_status(client, file_id)
    assert summary["status"] == "ready", summary
    assert summary["pages"] == 1

    file = client.get(f"/api/files/{file_id}").json()
    by_type = {}
    for f in file["findings"]:
        by_type.setdefault(f["type"], []).append(f)
    assert {"rut", "email", "phone", "name"} <= set(by_type)
    doubtful = [f for f in by_type["rut"] if f["doubtful"]]
    assert len(doubtful) == 1 and doubtful[0]["text"] == DOUBTFUL_RUT
    assert doubtful[0]["doubt_reason"] == "El dígito verificador no coincide: revisa el original"
    assert summary["counts"]["doubtful"] == 1
    for f in file["findings"]:
        assert f["status"] == "proposed" and f["history"][0]["action"] == "proposed"
        assert all(0 <= x <= 595 and 0 <= y <= 842 for x, y in f["polygon"])

    r = client.get(f"/api/files/{file_id}/pages/0.png?zoom=0.5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (298, 421)
    assert client.get(f"/api/files/{file_id}/pages/3.png").status_code == 404

    # The reviewer removes the doubtful RUT (it is a fictitious public code) with a reason.
    reason = "No es un dato personal"
    r = client.patch(
        f"/api/files/{file_id}/findings/{doubtful[0]['id']}",
        json={"action": "remove", "reason": reason, "note": "Código de proyecto inventado"},
    )
    assert r.status_code == 200 and r.json()["status"] == "removed"
    assert r.json()["history"][-1]["reason"] == reason
    again = client.patch(f"/api/files/{file_id}/findings/{doubtful[0]['id']}", json={"action": "remove"})
    assert again.status_code == 409 and again.json()["error"] == "already_removed"

    # And draws a zone over the title.
    r = client.post(
        f"/api/files/{file_id}/findings",
        json={"page": 0, "polygon": [[70, 85], [300, 85], [300, 105], [70, 105]], "note": "Título"},
    )
    assert r.status_code == 200
    manual = r.json()
    assert manual["type"] == "manual" and manual["status"] == "added" and manual["detector"] == "reviewer"
    assert manual["history"][0]["action"] == "added"
    bad = client.post(f"/api/files/{file_id}/findings", json={"page": 0, "polygon": [[1, 1], [1, 1], [1, 1]]})
    assert bad.status_code == 422 and bad.json()["message"]

    # Restore and remove again (restore puts "added" back for manual zones).
    r = client.patch(f"/api/files/{file_id}/findings/{manual['id']}", json={"action": "remove"})
    assert r.json()["status"] == "removed"
    r = client.patch(f"/api/files/{file_id}/findings/{manual['id']}", json={"action": "restore"})
    assert r.json()["status"] == "added"

    summary = next(f for f in client.get("/api/state").json()["files"] if f["id"] == file_id)
    assert summary["counts"]["removed"] == 1 and summary["counts"]["added"] == 1

    # Export is refused before confirmation.
    dest = tmp_path / "publicar"
    r = client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": True, "audit_json": True})
    assert r.status_code == 409 and r.json()["error"] == "nothing_to_export"

    r = client.post(f"/api/files/{file_id}/confirm")
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    assert client.post(f"/api/files/{file_id}/confirm").status_code == 409

    # An existing file with the same name is never overwritten.
    dest.mkdir()
    (dest / "informe_ficticio.pdf").write_bytes(b"no tocar")
    r = client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": True, "audit_json": True})
    assert r.status_code == 200, r.text
    body = r.json()
    result = body["results"][0]
    assert result["exported"] is True and result["leaks"] == []
    assert (dest / "informe_ficticio.pdf").read_bytes() == b"no tocar"
    out = Path(result["output_path"])
    assert out.name == "informe_ficticio (2).pdf"
    assert result["removed_by_reviewer"] == 1

    with pymupdf.open(out) as doc:
        text = "".join(page.get_text() for page in doc)
        assert VALID_RUT not in text and "12345678" not in text.replace(".", "")
        assert EMAIL not in text and "8123" not in text
        assert "Ana Prueba" not in text
        assert DOUBTFUL_RUT in text  # removed by the reviewer: stays visible
        assert "Informe ficticio" not in text  # covered by the manual zone
        assert not any(v for k, v in doc.metadata.items() if k not in ("format", "encryption"))
        assert not doc.get_xml_metadata()

    json_path, pdf_path = Path(body["audit"]["json_path"]), Path(body["audit"]["pdf_path"])
    assert json_path.is_file() and pdf_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    record = report["files"][0]
    assert record["leak_check"]["passed"] is True
    removal = [c for c in record["reviewer_changes"] if c["action"] == "removed" and c["type"] == "rut"]
    assert removal and removal[0]["reason"] == reason
    assert record["redactions_applied"] == len([f for f in record["findings"] if f["applied"]])
    raw = json_path.read_text(encoding="utf-8")
    assert VALID_RUT not in raw and EMAIL not in raw  # the report never contains censored data
    with pymupdf.open(pdf_path) as doc:
        # Expand typographic ligatures ("fi") the way PDF viewers do when searching.
        flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
        report_text = " ".join(" ".join(page.get_text(flags=flags) for page in doc).split())
    assert "Informe de auditoría de anonimización" in report_text
    assert reason in report_text
    assert "no certifica" in report_text
    assert VALID_RUT not in report_text

    state = client.get("/api/state").json()
    assert state["files"][0]["status"] == "exported"


def test_rotated_page_findings_in_view_space(client):
    file_id = upload(client, "rotado.pdf", make_pdf(rotation=90))
    client.post("/api/process", json={"file_ids": [file_id]})
    assert wait_status(client, file_id)["status"] == "ready"
    file = client.get(f"/api/files/{file_id}").json()
    page = file["pages"][0]
    assert (page["width"], page["height"]) == (842, 595)
    png = client.get(f"/api/files/{file_id}/pages/0.png").content
    assert Image.open(io.BytesIO(png)).size == (842, 595)
    rut = next(f for f in file["findings"] if f["type"] == "rut" and not f["doubtful"])
    xs = [p[0] for p in rut["polygon"]]
    ys = [p[1] for p in rut["polygon"]]
    # Rotated 90° clockwise, the text runs top to bottom: the box is tall and narrow.
    assert max(ys) - min(ys) > max(xs) - min(xs)
    assert all(0 <= x <= 842 for x in xs) and all(0 <= y <= 595 for y in ys)


def test_export_rotated_redacts(client, tmp_path):
    file_id = upload(client, "rotado.pdf", make_pdf(rotation=270))
    client.post("/api/process", json={"file_ids": [file_id]})
    assert wait_status(client, file_id)["status"] == "ready"
    client.post(f"/api/files/{file_id}/confirm")
    r = client.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": False, "audit_json": False})
    result = r.json()["results"][0]
    assert result["exported"], result
    assert r.json()["audit"] == {"json_path": None, "pdf_path": None}
    with pymupdf.open(result["output_path"]) as doc:
        assert VALID_RUT not in doc[0].get_text()


def test_errors_in_plain_spanish(client):
    ids = {
        "password": upload(client, "protegido.pdf", make_pdf(password="clave-inventada")),
        "corrupt": upload(client, "danado.pdf", b"%PDF-1.7\nesto no es un pdf"),
        "empty": upload(client, "vacio.pdf", b""),
        "format": upload(client, "notas.docx", b"PK\x03\x04 documento", "application/octet-stream"),
    }
    client.post("/api/process", json={})
    for code, file_id in ids.items():
        summary = wait_status(client, file_id)
        assert summary["status"] == "error", (code, summary)
        assert summary["error"] == code
        assert "Traceback" not in summary["error_message"]
    assert "contraseña" in wait_status(client, ids["password"])["error_message"]


def test_image_flow(client, tmp_path):
    img = Image.new("RGB", (400, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated: shown as 200 x 400
    exif[0x013B] = "Persona Inventada"  # Artist
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    file_id = upload(client, "foto.jpg", buf.getvalue(), "image/jpeg")
    r = client.post(f"/api/files/{file_id}/options", json={"all_text": True})
    assert r.json()["all_text"] is True
    client.post("/api/process", json={})
    summary = wait_status(client, file_id)
    assert summary["status"] == "ready" and summary["kind"] == "image"
    file = client.get(f"/api/files/{file_id}").json()
    assert (file["pages"][0]["width"], file["pages"][0]["height"]) == (200, 400)
    png = client.get(f"/api/files/{file_id}/pages/0.png").content
    assert Image.open(io.BytesIO(png)).size == (200, 400)
    client.post(
        f"/api/files/{file_id}/findings", json={"page": 0, "polygon": [[10, 10], [100, 10], [100, 60], [10, 60]]}
    )
    client.post(f"/api/files/{file_id}/confirm")
    r = client.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": True, "audit_json": True})
    result = r.json()["results"][0]
    assert result["exported"], result
    with Image.open(result["output_path"]) as out:
        assert out.size == (200, 400)
        assert len(out.getexif()) == 0
        assert out.convert("RGB").getpixel((50, 30))[0] < 30  # black
        assert out.convert("RGB").getpixel((150, 300))[0] > 220  # untouched


def test_exported_file_can_be_corrected_and_exported_again(client, tmp_path):
    file_id = upload(client, "ficticio.pdf", make_pdf())
    client.post("/api/process", json={"file_ids": [file_id]})
    wait_status(client, file_id)
    client.post(f"/api/files/{file_id}/confirm")
    first = client.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": False, "audit_json": False})
    assert first.json()["results"][0]["exported"]
    finding = client.get(f"/api/files/{file_id}").json()["findings"][0]
    r = client.patch(f"/api/files/{file_id}/findings/{finding['id']}", json={"action": "remove", "reason": "Otro"})
    assert r.status_code == 200, r.text
    summary = wait_status(client, file_id, statuses=("ready",))
    assert summary["step"] == "Listo para revisar"
    assert client.post(f"/api/files/{file_id}/confirm").status_code == 200
    second = client.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": False, "audit_json": False})
    result = second.json()["results"][0]
    assert result["exported"] and Path(result["output_path"]).name == "ficticio (2).pdf"
    assert (tmp_path / "ficticio.pdf").is_file()  # the first export is never overwritten


def test_cancel_and_delete(tmp_path):
    app = create_app(FakeEngine(delay=0.3), TOKEN)
    with LiveClient(app) as c:
        c.headers["X-Session-Token"] = TOKEN
        ids = [upload(c, f"doc{i}.pdf", make_pdf()) for i in range(3)]
        assert sorted(c.post("/api/process", json={}).json()["started"]) == sorted(ids)
        time.sleep(0.1)
        states = {f["id"]: f["status"] for f in c.get("/api/state").json()["files"]}
        assert list(states.values()).count("processing") <= 2  # at most two at a time
        assert c.post("/api/cancel", json={}).json() == {"ok": True}
        for file_id in ids:
            assert wait_status(c, file_id)["status"] == "cancelled"
        assert c.delete(f"/api/files/{ids[0]}").json() == {"ok": True}
        assert len(c.get("/api/state").json()["files"]) == 2
        r = c.get(f"/api/files/{ids[0]}")
        assert r.status_code == 404 and r.json()["error"] == "not_found"
        session_dir = app.state.session.dir
    assert not session_dir.exists()  # working copies are deleted when the session ends


def test_from_paths_folder(client, tmp_path):
    folder = tmp_path / "carpeta"
    (folder / "sub").mkdir(parents=True)
    original = make_pdf()
    (folder / "a.pdf").write_bytes(original)
    (folder / "sub" / "b.pdf").write_bytes(original)
    (folder / "notas.txt").write_text("ignorar")
    r = client.post("/api/files/from-paths", json={"paths": [str(folder), str(tmp_path / "no_existe.pdf")]})
    body = r.json()
    assert sorted(f["name"] for f in body["files"]) == ["a.pdf", "b.pdf"]
    assert body["skipped"][0]["message"] == "No se encontró este archivo."
    state = client.get("/api/state").json()["files"]
    assert {f["kind"] for f in state} == {"pdf"}
    assert (folder / "a.pdf").read_bytes() == original  # originals untouched


def test_leak_blocks_export(tmp_path):
    """A finding whose zone does not cover its text is caught by the leak check: nothing is written."""
    source = tmp_path / "work.pdf"
    source.write_bytes(make_pdf())
    engine = FakeEngine()
    file = AnalyzedFile(id="x1", name="fuga.pdf", path=str(source), kind="pdf")
    engine.analyze(file, [])
    rut = next(f for f in file.findings if f.type == "rut" and not f.doubtful)
    file.findings = [
        Finding(id="bad", file_id="x1", page=0, type="rut", polygon=[[0, 0], [5, 0], [5, 5], [0, 5]], text=rut.text)
    ]
    dest = tmp_path / "out"
    result = engine.export(file, str(dest))
    assert result.exported is False and result.output_path is None
    assert result.leaks and "RUT" in result.leaks[0].message
    assert not any(dest.iterdir())


def test_change_during_export_is_not_marked_exported(tmp_path):
    """A review change that arrives while the file is being exported reopens it: it is not marked exported."""

    class SlowEngine(FakeEngine):
        def export(self, file, dest_dir):
            result = super().export(file, dest_dir)
            file.status = "ready"  # what PATCH .../findings does to a confirmed file, meanwhile
            return result

    app = create_app(SlowEngine(), TOKEN)
    with LiveClient(app) as c:
        c.headers["X-Session-Token"] = TOKEN
        file_id = upload(c, "ficticio.pdf", make_pdf())
        c.post("/api/process", json={"file_ids": [file_id]})
        wait_status(c, file_id)
        c.post(f"/api/files/{file_id}/confirm")
        r = c.post("/api/export", json={"dest_dir": str(tmp_path), "audit_pdf": False, "audit_json": False})
        assert r.json()["results"][0]["exported"]
        assert wait_status(c, file_id, statuses=("ready", "exported"))["status"] == "ready"
