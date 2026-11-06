"""Local-only HTTP surface. No telemetry, no network calls at inference."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .api import Alternative, DecisionRequest, InvalidRequest, decide
from .select_read import SelectRead


def _payload(result) -> dict:
    return result.to_dict()


def make_handler(engine: SelectRead):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AzuriteLocal/1.0"

        def _send(self, status: int, obj: Any) -> None:
            body = json.dumps(obj, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "model": "azurite-select-read"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/decide":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                alternatives = [
                    Alternative(id=str(a["id"]), label=str(a["label"]),
                                description=a.get("description"))
                    for a in payload["alternatives"]
                ]
                request = DecisionRequest(
                    context=str(payload["context"]),
                    question=str(payload.get("question", "")),
                    alternatives=alternatives,
                    evidence_required=bool(payload.get("evidence_required", False)),
                    abstain_allowed=bool(payload.get("abstain_allowed", True)),
                )
                self._send(200, _payload(decide(request, engine)))
            except (InvalidRequest, KeyError, TypeError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):  # noqa: A003
            # Keep the local terminal quiet; no request data is logged.
            return

    return Handler


def serve(engine: SelectRead, *, host: str = "127.0.0.1", port: int = 8787) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"azurite local server on http://{host}:{port} (offline)")
    httpd.serve_forever()
