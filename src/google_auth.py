"""OAuth for Google's official remote MCP servers.

One-off authorization of an account (opens the browser, stores the refresh
token in ~/.secrets):

    uv run python -m src.google_auth <account> <preset>
    uv run python -m src.google_auth pro calendar

GoogleTokenAuth then keeps the access token fresh on every MCP request.
"""
import json
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_PORT = 8765
CLIENT_FILE = Path.home() / ".secrets/google-oauth-web.json"
TOKEN_DIR = Path.home() / ".secrets"

SCOPE_PRESETS = {
    "calendar": [
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/calendar.events.freebusy",
    ],
    "gmail": ["https://www.googleapis.com/auth/gmail.readonly"],
}


class GoogleTokenAuth(httpx.Auth):
    """Bearer auth backed by a stored refresh token; refreshes ahead of
    expiry so long-lived MCP connections never send a stale token."""

    def __init__(self, token_file):
        self.token_file = Path(token_file).expanduser()
        self._lock = threading.Lock()

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._access_token()}"
        yield request

    def _access_token(self):
        with self._lock:
            data = json.loads(self.token_file.read_text())
            if data.get("expires_at", 0) < time.time() + 60:
                data = self._refresh(data)
            return data["access_token"]

    def _refresh(self, data):
        response = httpx.post(TOKEN_URL, data={
            "client_id": data["client_id"],
            "client_secret": data["client_secret"],
            "refresh_token": data["refresh_token"],
            "grant_type": "refresh_token"})
        response.raise_for_status()
        fresh = response.json()
        data["access_token"] = fresh["access_token"]
        data["expires_at"] = time.time() + fresh.get("expires_in", 3600)
        self.token_file.write_text(json.dumps(data, indent=2))
        return data


def authorize(account, preset):
    client = json.loads(CLIENT_FILE.read_text())["web"]
    scopes = SCOPE_PRESETS[preset]
    redirect_uri = f"http://localhost:{REDIRECT_PORT}/"
    received = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            received["code"] = query.get("code", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("Autorisation reçue, tu peux fermer cet onglet.".encode())

        def log_message(self, *args):
            pass

    params = {"client_id": client["client_id"], "redirect_uri": redirect_uri,
              "response_type": "code", "scope": " ".join(scopes),
              "access_type": "offline", "prompt": "consent"}
    webbrowser.open(f"{AUTH_URL}?{urllib.parse.urlencode(params)}")
    print(f"En attente de l'autorisation dans le navigateur ({account}, {preset})...")
    server = HTTPServer(("localhost", REDIRECT_PORT), CallbackHandler)
    while not received.get("code"):
        server.handle_request()
    server.server_close()

    response = httpx.post(TOKEN_URL, data={
        "client_id": client["client_id"], "client_secret": client["client_secret"],
        "code": received["code"], "grant_type": "authorization_code",
        "redirect_uri": redirect_uri})
    response.raise_for_status()
    tokens = response.json()

    token_file = TOKEN_DIR / f"google-mcp-{account}-{preset}.json"
    token_file.write_text(json.dumps({
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "refresh_token": tokens["refresh_token"],
        "access_token": tokens["access_token"],
        "expires_at": time.time() + tokens.get("expires_in", 3600),
        "scopes": scopes}, indent=2))
    token_file.chmod(0o600)
    print(f"Token enregistré : {token_file}")


if __name__ == "__main__":
    import sys

    authorize(sys.argv[1], sys.argv[2])
