"""The HTTP-range zip reader used to pull single scenes out of the 12.5 GB Mip-NeRF 360 zip,
tested against a local server (no internet)."""
import http.server
import os
import sys
import threading
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import get_data  # noqa: E402


class RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""
    range_requests = 0

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.data)))
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng:
            type(self).range_requests += 1
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), len(self.data) - 1)
            body = self.data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        else:
            body = self.data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_extract_selected_members_over_http_ranges(tmp_path):
    zpath = tmp_path / "data.zip"
    files = {"garden/sparse/0/cameras.bin": os.urandom(5000), "garden/images_4/a.png": os.urandom(300_000),
             "garden/images/a.png": os.urandom(2_000_000), "bicycle/images_4/b.png": os.urandom(1000)}
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for k, v in files.items():
            z.writestr(k, v)
    RangeHandler.data = zpath.read_bytes()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/data.zip"
        f = get_data.HTTPRangeFile(url, block=64 * 1024)
        assert f.size == len(RangeHandler.data)
        keep = lambda m: m.startswith("garden/sparse/") or m.startswith("garden/images_4/")
        out = tmp_path / "out"
        n = get_data.extract_members(url, str(out), keep)
        assert n == 2
        assert (out / "garden/sparse/0/cameras.bin").read_bytes() == files["garden/sparse/0/cameras.bin"]
        assert (out / "garden/images_4/a.png").read_bytes() == files["garden/images_4/a.png"]
        assert not (out / "garden/images/a.png").exists() and not (out / "bicycle").exists()
        assert RangeHandler.range_requests > 0
    finally:
        srv.shutdown()
