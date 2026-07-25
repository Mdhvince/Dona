import json
import time

import httpx

from src.google_auth import GoogleTokenAuth


def token_file(tmp_path, expires_at):
    path = tmp_path / "token.json"
    path.write_text(json.dumps({
        "client_id": "id", "client_secret": "secret",
        "refresh_token": "refresh", "access_token": "old",
        "expires_at": expires_at}))
    return path


def bearer_of(auth):
    request = httpx.Request("GET", "https://example.com")
    return next(auth.auth_flow(request)).headers["Authorization"]


def test_valid_token_is_used_without_refresh(tmp_path, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("refresh must not happen")

    monkeypatch.setattr(httpx, "post", never)
    auth = GoogleTokenAuth(token_file(tmp_path, time.time() + 3600))
    assert bearer_of(auth) == "Bearer old"


def test_expired_token_is_refreshed_and_persisted(tmp_path, monkeypatch):
    def fake_post(url, data):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "refresh"
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    path = token_file(tmp_path, time.time() - 10)
    assert bearer_of(GoogleTokenAuth(path)) == "Bearer new"
    saved = json.loads(path.read_text())
    assert saved["access_token"] == "new"
    assert saved["expires_at"] > time.time() + 3000
