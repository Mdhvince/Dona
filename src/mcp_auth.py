"""OAuth for remote MCP servers, using the protocol's own flow (metadata
discovery, dynamic client registration, PKCE, refresh) from the MCP SDK.

One-off authorization of a server declared in config.toml:

    uv run python -m src.mcp_auth qonto

Tokens and client registration land in ~/.secrets, one file per server, and
are refreshed automatically afterwards.
"""
import asyncio
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

CALLBACK_PORT = 8765
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"
TOKEN_DIR = Path.home() / ".secrets"


class FileTokenStorage(TokenStorage):
    """Client registration and tokens on disk, one file per server."""

    def __init__(self, name):
        self.path = TOKEN_DIR / f"mcp-{name}-oauth.json"

    def _read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _write(self, data):
        TOKEN_DIR.mkdir(mode=0o700, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        self.path.chmod(0o600)

    async def get_tokens(self):
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens):
        self._write({**self._read(), "tokens": tokens.model_dump(mode="json")})

    async def get_client_info(self):
        raw = self._read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info):
        self._write({**self._read(), "client": client_info.model_dump(mode="json")})


def _wait_for_callback():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            received["code"] = query.get("code", [""])[0]
            received["state"] = query.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("Autorisation reçue, tu peux fermer cet onglet.".encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("localhost", CALLBACK_PORT), Handler)
    while not received.get("code"):
        server.handle_request()
    server.server_close()
    return received["code"], received.get("state")


def oauth_provider(name, server_url):
    """httpx auth handling the whole MCP OAuth flow; the browser only opens
    when no valid token is stored."""

    async def redirect_handler(url):
        print(f"Autorise l'accès à {name} dans le navigateur...")
        webbrowser.open(url)

    async def callback_handler():
        return await asyncio.to_thread(_wait_for_callback)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=OAuthClientMetadata(
            client_name="Dona",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"]),
        storage=FileTokenStorage(name),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler)


if __name__ == "__main__":
    import sys

    from src.config import load_config
    from src.tools import load_mcp_tools

    name = sys.argv[1]
    config = load_config()
    config["mcp"] = [s for s in config["mcp"] if s["name"] == name]
    if not config["mcp"]:
        raise SystemExit(f"Serveur {name} absent de config.toml")
    # Whitelist dropped on purpose: the authorization run lists everything
    # the server exposes, so the read-only selection can be made informed
    config["mcp"][0].pop("tools", None)
    for tool in load_mcp_tools(config):
        print(f"  {tool.name}")
