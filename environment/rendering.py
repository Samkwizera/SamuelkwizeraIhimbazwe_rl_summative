import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).parent / "web"


class _Handler(SimpleHTTPRequestHandler):
    renderer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/frames":
            return super().do_GET()
        since = int(parse_qs(url.query).get("since", ["0"])[0])
        self.renderer.connected.set()
        with self.renderer.lock:
            frames = self.renderer.frames[since:]
        body = json.dumps(frames).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keeps the terminal free for the agent's own output


class ThreeJSRenderer:
    """Serves the three.js village page and streams every env step to it."""

    def __init__(self, port=8765, frame_delay=0.6, open_browser=True):
        self.frames = []
        self.lock = threading.Lock()
        self.connected = threading.Event()
        self.frame_delay = frame_delay
        self.open_browser = open_browser

        handler = type("Handler", (_Handler,), {"renderer": self})
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._opened = False

    def update(self, snapshot):
        with self.lock:
            snapshot["frame"] = len(self.frames)
            self.frames.append(snapshot)
        if not self._opened:
            self._opened = True
            print(f"3D view running at {self.url}")
            if self.open_browser:
                webbrowser.open(self.url)
            # don't start stepping until the page is listening, otherwise the video misses the start
            self.connected.wait(timeout=20)
        time.sleep(self.frame_delay)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
