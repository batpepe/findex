"""A local mirror of the Python HTML docs with simulated network latency (and failures), for Lab 6.

    ./scripts/get-mirror.sh                                  # once: docs.python.org's HTML archive
    uv run python scripts/mirror_server.py mirror --port 8765 --latency 0.15 [--fail-rate 0.05]

Every response waits ``--latency`` seconds first (the median page time I
measured to docs.python.org was ~0.26 s), so concurrency can be measured
without hammering anyone's server. ``--fail-rate`` answers that share of first
requests with ``503 Retry-After: 1``, to watch the crawler's retries.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import ClassVar


class QuickHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:  # skip HTTPServer's reverse-DNS getfqdn() (30 s on some Macs)
        TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", int(self.server_address[1])


class MirrorHandler(SimpleHTTPRequestHandler):
    latency = 0.15
    fail_rate = 0.0
    _failed: ClassVar[set[str]] = set()  # paths already failed once, shared by all handler threads
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def do_GET(self) -> None:
        time.sleep(self.latency)  # a thread per request: this models the network, not the server
        with self._lock:
            fail = self.path not in self._failed and random.random() < self.fail_rate
            if fail:
                self._failed.add(self.path)
        if fail:
            self.send_response(503)
            self.send_header("Retry-After", "1")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--latency", type=float, default=0.15)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    args = parser.parse_args()
    MirrorHandler.latency, MirrorHandler.fail_rate = args.latency, args.fail_rate
    server = QuickHTTPServer(("127.0.0.1", args.port), partial(MirrorHandler, directory=args.root))
    print(f"serving {args.root} on http://127.0.0.1:{args.port}/ (latency {args.latency}s)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
