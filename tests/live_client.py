"""A small HTTP client for tests that talks to the real local server (uvicorn on 127.0.0.1).

It only uses the standard library, so the tests need no extra HTTP package, and it exercises
the same server, middleware and lifespan as the desktop app.
"""

from __future__ import annotations

import json as jsonlib
import urllib.error
import urllib.request
import uuid
from http.cookiejar import CookieJar

from anonymizer.app import Server, bind_socket


class Response:
    def __init__(self, status: int, headers, content: bytes):
        self.status_code = status
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return jsonlib.loads(self.content)


def _multipart(files: list[tuple[str, tuple[str, bytes, str]]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for field, (name, data, mime) in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class LiveClient:
    def __init__(self, app):
        self.server = Server(app, bind_socket())
        self.base = f"http://127.0.0.1:{self.server.port}"
        self.headers: dict[str, str] = {}
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def __enter__(self) -> LiveClient:
        self.server.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.stop()

    def request(self, method: str, path: str, *, json=None, files=None, headers=None) -> Response:
        data, all_headers = None, {**self.headers, **(headers or {})}
        if json is not None:
            data = jsonlib.dumps(json).encode()
            all_headers["Content-Type"] = "application/json"
        elif files is not None:
            data, all_headers["Content-Type"] = _multipart(files)
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=all_headers)
        try:
            with self.opener.open(req, timeout=60) as resp:
                return Response(resp.status, resp.headers, resp.read())
        except urllib.error.HTTPError as err:
            return Response(err.code, err.headers, err.read())

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def put(self, path, **kw):
        return self.request("PUT", path, **kw)

    def patch(self, path, **kw):
        return self.request("PATCH", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)
