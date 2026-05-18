#!/usr/bin/env python3
"""Static server for the V2V debug dashboard.

Run anywhere — laptop, dev box, device itself. Backend is reached over
WebSocket via the URL configured in the dashboard UI, not via this server.

    uv run tools/v2v-debug/serve.py
    python3 tools/v2v-debug/serve.py --port 8080
"""
import argparse
import http.server
import pathlib
import socketserver
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default 0.0.0.0 so LAN can reach)")
    ap.add_argument("--no-open", action="store_true",
                    help="do not auto-open a browser tab")
    args = ap.parse_args()

    root = pathlib.Path(__file__).parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}/"
        print(f"V2V debug dashboard → {url}  (Ctrl+C to stop)")
        if not args.no_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
