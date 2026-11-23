"""Reader for COLMAP sparse models (cameras / images / points3D, binary or text).

Formats as documented at https://colmap.github.io/format.html. COLMAP stores the
world->camera rotation as a quaternion (qw, qx, qy, qz) and the translation t,
in the same axis convention as ours (x right, y down, z forward), and its pixel
coordinates put the centre of the top-left pixel at (0.5, 0.5), as ours do.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

# model_id -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: (mid, n) for mid, (name, n) in CAMERA_MODELS.items()}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy); only undistorted pinhole models are accepted."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        if self.model == "PINHOLE":
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        raise ValueError(f"camera model {self.model} has distortion; undistort the images first "
                         f"(colmap image_undistorter)")


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray      # (4,) w, x, y, z  world -> camera
    tvec: np.ndarray      # (3,)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def rotmat(self) -> np.ndarray:
        w, x, y, z = self.qvec / np.linalg.norm(self.qvec)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    xyz: np.ndarray          # (P, 3)
    rgb: np.ndarray          # (P, 3) uint8
    point_ids: np.ndarray    # (P,)


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _read(f, "iiQQ")
            name, k = CAMERA_MODELS[mid]
            cams[cid] = ColmapCamera(cid, name, w, h, np.array(_read(f, "d" * k)))
    return cams


def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = _read(f, "Q")
            data = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            imgs[iid] = ColmapImage(iid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid,
                                    name.decode("utf-8"), np.stack([data["x"], data["y"]], -1),
                                    data["id"].astype(np.int64))
    return imgs


def read_points3d_binary(path):
    ids, xyz, rgb = [], [], []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            pid, x, y, z, r, g, b, _err, tlen = _read(f, "QdddBBBdQ")
            f.seek(8 * tlen, 1)  # track: (int32 image_id, int32 point2D_idx) per element
            ids.append(pid)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return (np.array(ids, dtype=np.int64), np.array(xyz, dtype=np.float64).reshape(-1, 3),
            np.array(rgb, dtype=np.uint8).reshape(-1, 3))


def _data_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path):
    cams = {}
    for line in _data_lines(path):
        el = line.split()
        cid = int(el[0])
        cams[cid] = ColmapCamera(cid, el[1], int(el[2]), int(el[3]), np.array([float(v) for v in el[4:]]))
    return cams


def read_images_text(path):
    imgs = {}
    # the points line of an image may be empty, so blank lines matter here: read raw
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        el = lines[i].split()
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        arr = np.array([float(v) for v in pts]).reshape(-1, 3)
        iid = int(el[0])
        imgs[iid] = ColmapImage(iid, np.array([float(v) for v in el[1:5]]), np.array([float(v) for v in el[5:8]]),
                                int(el[8]), " ".join(el[9:]), arr[:, :2], arr[:, 2].astype(np.int64))
        i += 2
    return imgs


def read_points3d_text(path):
    ids, xyz, rgb = [], [], []
    for line in _data_lines(path):
        el = line.split()
        ids.append(int(el[0]))
        xyz.append([float(v) for v in el[1:4]])
        rgb.append([int(v) for v in el[4:7]])
    return (np.array(ids, dtype=np.int64), np.array(xyz, dtype=np.float64).reshape(-1, 3),
            np.array(rgb, dtype=np.uint8).reshape(-1, 3))


def read_model(sparse_dir: str) -> ColmapModel:
    """Read a model from `sparse_dir` (e.g. scene/sparse/0), preferring binary files."""
    if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        cams = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        imgs = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        ids, xyz, rgb = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))
    else:
        cams = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        imgs = read_images_text(os.path.join(sparse_dir, "images.txt"))
        ids, xyz, rgb = read_points3d_text(os.path.join(sparse_dir, "points3D.txt"))
    return ColmapModel(cams, imgs, xyz, rgb, ids)


def reprojection_errors(model: ColmapModel) -> np.ndarray:
    """Pixel distance between each observed keypoint and its 3D point projected with our
    reading of the poses and intrinsics. Small (~1 px) errors mean the conventions are right."""
    index = {int(pid): i for i, pid in enumerate(model.point_ids)}
    errs = []
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = cam.pinhole()
        R, t = im.rotmat(), im.tvec
        for xy, pid in zip(im.xys, im.point3d_ids):
            if pid < 0 or int(pid) not in index:
                continue
            pc = R @ model.xyz[index[int(pid)]] + t
            uv = np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy])
            errs.append(np.linalg.norm(uv - xy))
    return np.array(errs)
