"""Signing in to a portal that hides records behind NextAuth + Keycloak."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from dcat_ap_hub import Dataset
from dcat_ap_hub.metadata import auth

USERNAME, PASSWORD = "alice", "test-only-password"


class FakePortal(BaseHTTPRequestHandler):
    """A NextAuth portal and its Keycloak, as green.data.sdil.de and piveau.hlrs.de."""

    received_passwords: list = []

    def _signed_in(self):
        return "session=ok" in (self.headers.get("Cookie") or "")

    def _send(self, status, body=b"", content_type="text/html", headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, headers=()):
        self._send(200, json.dumps(value).encode(), "application/json", headers)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if path in ("/record.jsonld", "/files/data.csv"):
            if not self._signed_in():
                return self._send(307, headers=[("Location", f"/auth/signin?callbackUrl={path}")])
            if path == "/files/data.csv":
                return self._send(200, b"x\n1\n", "text/csv")
            return self._json({"@graph": [
                {"@id": f"{base}/d", "@type": "dcat:Dataset", "dct:title": "Protected"},
                {"@type": "dcat:Distribution", "dct:title": "data.csv", "dcat:downloadURL": f"{base}/files/data.csv"},
            ]})
        if path == "/auth/signin":
            return self._send(200, b"<html>Sign in</html>")
        if path == "/api/auth/providers":
            return self._json({"keycloak": {
                "id": "keycloak", "type": "oauth", "signinUrl": f"{base}/api/auth/signin/keycloak",
            }})
        if path == "/api/auth/csrf":
            return self._json({"csrfToken": "csrf-1"})
        if path == "/kc/auth":
            form = (
                '<form id="kc-form-login" method="post" '
                'action="/kc/login?session_code=abc&amp;tab_id=1">'
                '<input name="username"><input name="password" type="password">'
                '<input type="hidden" name="credentialId" value=""></form>'
            )
            return self._send(200, form.encode())
        if path == "/api/auth/callback/keycloak":
            return self._send(302, headers=[("Set-Cookie", "session=ok; Path=/"), ("Location", "/")])
        if path == "/api/auth/session":
            return self._json({"user": {"name": USERNAME}} if self._signed_in() else {})
        if path == "/":
            return self._send(200, b"<html>home</html>")
        self._send(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode())
        path = urlparse(self.path).path
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if path == "/api/auth/signin/keycloak":
            assert form["csrfToken"] == ["csrf-1"]
            return self._json({"url": f"{base}/kc/auth?client_id=piveau-hub-ui"})
        if path == "/kc/login":
            FakePortal.received_passwords.append(form.get("password", [""])[0])
            assert parse_qs(urlparse(self.path).query)["tab_id"] == ["1"]
            if form.get("username") == [USERNAME] and form.get("password") == [PASSWORD]:
                return self._send(302, headers=[("Location", "/api/auth/callback/keycloak?code=1")])
            return self._send(200, b'<form id="kc-form-login" action="/kc/login"></form>Invalid')
        self._send(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def portal(tmp_path, monkeypatch):
    # No real .env must leak in; and no session may survive between tests.
    monkeypatch.chdir(tmp_path)
    for name in (auth.LOGIN_HOSTS_ENV, auth.USERNAME_ENV, auth.PASSWORD_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(auth, "_sessions", {})
    FakePortal.received_passwords = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePortal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_without_login_settings_the_error_explains_what_to_set(portal):
    with pytest.raises(auth.LoginRequired, match="DCAT_AP_HUB_LOGIN_HOSTS=127.0.0.1"):
        Dataset.from_url(f"{portal}/record.jsonld")


def test_credentials_are_not_sent_to_portals_that_are_not_listed(portal, monkeypatch):
    monkeypatch.setenv(auth.LOGIN_HOSTS_ENV, "green.data.sdil.de")
    monkeypatch.setenv(auth.USERNAME_ENV, USERNAME)
    monkeypatch.setenv(auth.PASSWORD_ENV, PASSWORD)

    with pytest.raises(auth.LoginRequired):
        Dataset.from_url(f"{portal}/record.jsonld")
    assert FakePortal.received_passwords == []


def test_signs_in_and_downloads_protected_record_and_files(portal, tmp_path, monkeypatch):
    # Settings may come from .env in the working directory.
    (tmp_path / ".env").write_text(
        f"{auth.LOGIN_HOSTS_ENV}=127.0.0.1\n{auth.USERNAME_ENV}={USERNAME}\n"
        f'{auth.PASSWORD_ENV}="{PASSWORD}"\n'
    )

    ds = Dataset.from_url(f"{portal}/record.jsonld")
    files = ds.download(data_dir=tmp_path / "data", verbose=False)

    assert ds.title == "Protected"
    [table] = files.filter_by("csv")
    assert table.path.read_text() == "x\n1\n"
    assert (files.root / "dcat-ap-metadata.jsonld").is_file()
    # One sign-in serves the record, the saved sidecar, and the download.
    assert FakePortal.received_passwords == [PASSWORD]


def test_settings_come_from_the_nearest_dotenv_in_a_parent(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DCAT_AP_HUB_TEST_SETTING=from-parent\n")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    monkeypatch.delenv("DCAT_AP_HUB_TEST_SETTING", raising=False)

    assert auth.setting("DCAT_AP_HUB_TEST_SETTING") == "from-parent"


def test_wrong_password_fails_clearly(portal, monkeypatch):
    monkeypatch.setenv(auth.LOGIN_HOSTS_ENV, "127.0.0.1")
    monkeypatch.setenv(auth.USERNAME_ENV, USERNAME)
    monkeypatch.setenv(auth.PASSWORD_ENV, "wrong")

    with pytest.raises(auth.LoginFailed, match="wrong username or password"):
        Dataset.from_url(f"{portal}/record.jsonld")


def test_non_json_answers_get_a_clear_error(portal):
    with pytest.raises(ValueError, match="did not return JSON-LD metadata but text/html"):
        Dataset.from_url(f"{portal}/")
