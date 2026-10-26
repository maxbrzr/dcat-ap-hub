"""HTTP access to catalog portals, including ones that hide records behind a login.

Some piveau portals (e.g. ``green.data.sdil.de``) redirect anonymous requests
to ``/auth/signin``: a NextAuth front end that signs users in through
Keycloak. dcat-ap-hub can sign in there the way a browser does, with a
username and password, but only for portals the user lists explicitly:

* ``DCAT_AP_HUB_LOGIN_HOSTS``: comma-separated portal hosts that may receive
  the credentials, e.g. ``green.data.sdil.de``;
* ``DCAT_AP_HUB_USERNAME`` and ``DCAT_AP_HUB_PASSWORD``.

Each is read from the environment or the nearest ``.env`` file (in the
working directory or a parent). Without them, such a portal raises :class:`LoginRequired`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

LOGIN_HOSTS_ENV = "DCAT_AP_HUB_LOGIN_HOSTS"
USERNAME_ENV = "DCAT_AP_HUB_USERNAME"
PASSWORD_ENV = "DCAT_AP_HUB_PASSWORD"
_SIGNIN_PATH = "/auth/signin"
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# One session per portal host, so a sign-in is reused for every request.
_sessions: Dict[str, requests.Session] = {}


class LoginRequired(RuntimeError):
    """The portal requires sign-in and dcat-ap-hub has no permission to sign in."""


class LoginFailed(RuntimeError):
    """Signing in to the portal did not produce a session."""


def setting(name: str, dotenv_path: Optional[Path] = None) -> str:
    """
    Return a setting from the environment, else from the nearest ``.env``.

    The nearest ``.env`` is in the working directory or one of its parents,
    so a notebook in ``notebooks/`` finds the project's ``.env``.
    """
    value = os.getenv(name)
    if value:
        return value
    dotenv_path = dotenv_path or _nearest_dotenv()
    if dotenv_path is None:
        return ""
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, sep, raw = line.strip().partition("=")
        if not sep or key.strip() != name or line.lstrip().startswith("#"):
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        return raw
    return ""


def _nearest_dotenv() -> Optional[Path]:
    for folder in (Path.cwd(), *Path.cwd().parents):
        if (folder / ".env").is_file():
            return folder / ".env"
    return None


def get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    stream: bool = False,
    timeout: float = 60,
) -> requests.Response:
    """
    ``GET`` a URL, signing in first if its portal requires it and may be signed in to.

    Raises :class:`LoginRequired` when the portal demands a login that is not
    configured, and :class:`LoginFailed` when the configured login fails.
    """
    parsed = urlparse(url)
    session = _sessions.setdefault(parsed.netloc, requests.Session())
    response = session.get(url, headers=headers, stream=stream, timeout=timeout)
    if not _redirected_to_signin(response):
        return response
    response.close()

    host = parsed.hostname or ""
    allowed = {h.strip() for h in setting(LOGIN_HOSTS_ENV).split(",") if h.strip()}
    username, password = setting(USERNAME_ENV), setting(PASSWORD_ENV)
    if host not in allowed or not (username and password):
        raise LoginRequired(
            f"{url} requires signing in at {host}. To let dcat-ap-hub sign in, set "
            f"{LOGIN_HOSTS_ENV}={host}, {USERNAME_ENV}, and {PASSWORD_ENV} in the "
            "environment or in .env."
        )
    _sign_in(session, f"{parsed.scheme}://{parsed.netloc}", username, password, timeout)
    response = session.get(url, headers=headers, stream=stream, timeout=timeout)
    if _redirected_to_signin(response):
        response.close()
        raise LoginFailed(f"Signed in to {host}, but {url} still requires signing in.")
    return response


def _redirected_to_signin(response: requests.Response) -> bool:
    return urlparse(response.url).path.endswith(_SIGNIN_PATH)


def _sign_in(
    session: requests.Session, base: str, username: str, password: str, timeout: float
) -> None:
    """Sign in to a NextAuth portal through its OAuth (Keycloak) provider."""
    providers = session.get(f"{base}/api/auth/providers", timeout=timeout).json()
    provider = next(
        (p for p in providers.values() if p.get("type") in {"oauth", "oidc"}), None
    )
    if provider is None:
        raise LoginFailed(f"{base} offers no OAuth sign-in dcat-ap-hub can use.")
    csrf = session.get(f"{base}/api/auth/csrf", timeout=timeout).json()["csrfToken"]
    started = session.post(
        provider["signinUrl"],
        data={"csrfToken": csrf, "callbackUrl": f"{base}/", "json": "true"},
        allow_redirects=False,
        timeout=timeout,
    )
    authorize_url = (
        started.json().get("url")
        if "json" in started.headers.get("Content-Type", "")
        else started.headers.get("Location")
    )
    if not authorize_url:
        raise LoginFailed(f"{base} did not start a sign-in.")

    login_page = session.get(authorize_url, timeout=timeout)
    form = BeautifulSoup(login_page.text, "html.parser").find("form", id="kc-form-login")
    if form is None or not form.get("action"):
        raise LoginFailed(f"No username/password form at {urlparse(login_page.url).netloc}.")
    action = urljoin(login_page.url, form["action"])
    target = urlparse(action)
    # The password goes only to the identity provider the trusted portal chose,
    # and only over HTTPS (plain HTTP is allowed for local test servers).
    if target.netloc != urlparse(login_page.url).netloc or (
        target.scheme != "https" and target.hostname not in _LOCAL_HOSTS
    ):
        raise LoginFailed(f"Refusing to send the password to {action}.")
    fields = {
        field["name"]: field.get("value", "")
        for field in form.find_all("input")
        if field.get("name")
    }
    fields.update({"username": username, "password": password})
    session.post(action, data=fields, timeout=timeout)

    signed_in = session.get(f"{base}/api/auth/session", timeout=timeout).json()
    if not signed_in or not signed_in.get("user"):
        raise LoginFailed(
            f"Signing in to {urlparse(base).hostname} failed: wrong username or "
            "password, or an extra step (such as two-factor authentication) that "
            "dcat-ap-hub cannot complete."
        )
