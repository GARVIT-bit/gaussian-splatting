"""Download the evaluation datasets (run in the Kaggle/Colab session, never on the laptop).

    python scripts/get_data.py blender    --dest /tmp/data --scenes all
    python scripts/get_data.py mipnerf360 --dest /tmp/data --scenes garden bicycle kitchen
    python scripts/get_data.py tandt      --dest /tmp/data --scenes truck

Sources:
  * NeRF-synthetic lego: the NeRF authors' example zip (official).
  * NeRF-synthetic other scenes: the nerfbaselines mirror on Hugging Face (one zip per scene);
    the mirror's lego is compared file by file with the official lego before it is trusted.
  * Mip-NeRF 360: the official 360_v2.zip on Google storage. Only the requested scenes'
    sparse/ model and their evaluation image folder (images_4 outdoor, images_2 indoor, as in
    the official 3DGS full_eval.py) are read, with HTTP range requests, not the full 12.5 GB.
  * Tanks & Temples (truck, train): the 3DGS authors' release tandt_db.zip (COLMAP included).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

NERF_EXAMPLE = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
BLENDER_MIRROR = "https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/main/blender/{}.zip"
MIP360 = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
TANDT = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
BLENDER_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
MIP_OUTDOOR = ["bicycle", "flowers", "garden", "stump", "treehill"]
MIP_INDOOR = ["room", "counter", "kitchen", "bonsai"]


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, so zipfile can read single members."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache_start, self._cache = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):  # inclusive end
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if r.status != 206 and not (start == 0 and len(data) == self.size):
            raise IOError(f"server ignored the range request (HTTP {r.status})")
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0 = self._cache_start
        if not (c0 >= 0 and c0 <= self.pos and self.pos + n <= c0 + len(self._cache)):
            start = self.pos
            end = min(self.size, start + max(n, self.block)) - 1
            self._cache_start, self._cache = start, self._fetch(start, end)
        off = self.pos - self._cache_start
        out = self._cache[off:off + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def extract_members(url_or_path: str, dest: str, keep) -> int:
    """Extract the members of a (possibly remote) zip for which keep(name) is true."""
    src = HTTPRangeFile(url_or_path) if url_or_path.startswith("http") else open(url_or_path, "rb")
    n = 0
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if info.is_dir() or not keep(info.filename):
                continue
            target = os.path.join(dest, info.filename)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                n += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            n += 1
    return n


def download(url: str, path: str):
    if os.path.exists(path):
        return path
    print("downloading", url, flush=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: str) -> dict[str, str]:
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = sha256(p)
    return out


def get_blender(dest: str, scenes: list[str]):
    base = os.path.join(dest, "nerf_synthetic")
    os.makedirs(base, exist_ok=True)
    # official lego
    official = os.path.join(dest, "official_lego")
    lego_done = os.path.exists(os.path.join(base, "lego", "transforms_test.json"))
    if "lego" in scenes and not lego_done and not os.path.exists(os.path.join(official, "transforms_test.json")):
        zp = download(NERF_EXAMPLE, os.path.join(dest, "nerf_example_data.zip"))
        n = extract_members(zp, os.path.join(dest, "_official"), lambda m: "nerf_synthetic/lego/" in m)
        src = os.path.join(dest, "_official", "nerf_synthetic", "lego")
        shutil.move(src, official)
        shutil.rmtree(os.path.join(dest, "_official"), ignore_errors=True)
        os.remove(zp)
        print(f"official lego: {n} files")
    # mirror
    for s in scenes:
        if os.path.exists(os.path.join(base, s, "transforms_test.json")):
            continue
        zp = download(BLENDER_MIRROR.format(s), os.path.join(dest, f"{s}.zip"))
        tmp = os.path.join(dest, f"_{s}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(tmp)
        os.remove(zp)
        found = [d for d, _, fs in os.walk(tmp) if "transforms_test.json" in fs]
        assert len(found) == 1, f"{s}: unexpected zip layout {found}"
        shutil.move(found[0], os.path.join(base, s))
        shutil.rmtree(tmp, ignore_errors=True)
        print("scene", s, "ok", flush=True)
    # provenance check: mirror lego must equal the official lego, file by file
    if "lego" in scenes and os.path.exists(official):
        a, b = tree_hashes(official), tree_hashes(os.path.join(base, "lego"))
        common = sorted(set(a) & set(b))
        same = sum(a[k] == b[k] for k in common)
        print(f"mirror vs official lego: {len(a)} official files, {len(b)} mirror files, {len(common)} in common, "
              f"{same} identical")
        if len(common) < 0.9 * len(a) or same != len(common):
            raise SystemExit("the mirror's lego differs from the official lego: not trusting the mirror")
        shutil.rmtree(os.path.join(base, "lego"))
        shutil.move(official, os.path.join(base, "lego"))


def get_mipnerf360(dest: str, scenes: list[str]):
    base = os.path.join(dest, "360_v2")
    for s in scenes:
        folder = "images_4" if s in MIP_OUTDOOR else "images_2"
        keep = lambda m, s=s, folder=folder: m.startswith(f"{s}/sparse/") or m.startswith(f"{s}/{folder}/")
        n = extract_members(MIP360, base, keep)
        print(f"360_v2/{s}: {n} files ({folder} + sparse)", flush=True)
        assert n > 10, f"{s}: nothing extracted; zip layout changed?"


def get_tandt(dest: str, scenes: list[str]):
    zp = download(TANDT, os.path.join(dest, "tandt_db.zip"))
    n = extract_members(zp, dest, lambda m: any(m.startswith(f"tandt/{s}/") for s in scenes))
    print(f"tandt: {n} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["blender", "mipnerf360", "tandt"])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--scenes", nargs="+", default=["all"])
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    if a.dataset == "blender":
        get_blender(a.dest, BLENDER_SCENES if a.scenes == ["all"] else a.scenes)
    elif a.dataset == "mipnerf360":
        get_mipnerf360(a.dest, MIP_OUTDOOR + MIP_INDOOR if a.scenes == ["all"] else a.scenes)
    else:
        get_tandt(a.dest, ["truck", "train"] if a.scenes == ["all"] else a.scenes)


if __name__ == "__main__":
    sys.exit(main())
