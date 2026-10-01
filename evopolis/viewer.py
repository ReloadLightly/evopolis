"""Local community replay: python -m evopolis.viewer --port 8765."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .viewer_data import catalog, get_episode, prepare_cache, sandbox
from .trained_viewer_data import generated_catalog, generated_episode

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).with_name("static")
ASSETS = {"/": (STATIC / "index.html", "text/html"),
          "/app.js": (STATIC / "app.js", "text/javascript"),
          "/town.js": (STATIC / "town.js", "text/javascript"),
          "/style.css": (STATIC / "style.css", "text/css"),
          "/theme.css": (ROOT / "docs/assets/evopolis-theme.css", "text/css")}


def make_handler(cache):
    trained = generated_catalog()
    trained_ids = {episode["id"] for episode in trained["episodes"]}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            data = json.dumps(body, allow_nan=False).encode() if content_type == "application/json" else body
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/catalog":
                    recorded = catalog(cache)
                    self.respond(200, {**recorded,
                        "episodes": recorded["episodes"] + trained["episodes"],
                        "trained_default_id": trained["default_id"],
                        "trained_provenance": trained["provenance"]})
                elif url.path == "/api/episode":
                    identity = parse_qs(url.query).get("id", [""])[0]
                    self.respond(200, generated_episode(identity) if identity in trained_ids else get_episode(cache, identity))
                elif url.path in ASSETS:
                    path, mime = ASSETS[url.path]
                    self.respond(200, path.read_bytes(), mime)
                elif url.path == "/favicon.ico":
                    self.respond(204, b"", "image/x-icon")
                else:
                    self.respond(404, {"error": "Resource not found"})
            except (KeyError, ValueError) as exc:
                self.respond(404, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/api/sandbox":
                return self.respond(404, {"error": "Resource not found"})
            # This is a local read/compute service; it accepts no file paths or code.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Expected a small JSON settings object")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a settings object")
                self.respond(200, sandbox(body.get("mechanism"), body.get("fractions")))
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    cache = prepare_cache(ROOT / "data/raw", ROOT / "data/cache/viewer")
    if args.prepare_only:
        return
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cache))
    except OSError as exc:
        parser.exit(1, f"Cannot bind 127.0.0.1:{args.port}: {exc}. Try --port {args.port + 1}.\n")
    print(f"EvoPolis community viewer: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
