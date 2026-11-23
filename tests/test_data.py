"""Loaders: Blender (NeRF-synthetic) and COLMAP, on small datasets written by the tests."""
import json
import math
import os
import struct

import numpy as np
import torch
from PIL import Image

from splat.camera import look_at, orbit_cameras
from splat.colmap import MODEL_IDS, read_model, reprojection_errors
from splat.data import camera_extent, load_blender, load_colmap
from splat.gaussians import Gaussians
from splat.reference import render


# ------------------------------------------------------------------ Blender

def write_blender_dataset(root, cams_by_split, gt: Gaussians, fov_x_deg: float):
    """Render `gt` from OpenCV cameras and save them in the NeRF-synthetic format (RGBA PNG,
    camera->world in OpenGL axes)."""
    for split, cams in cams_by_split.items():
        os.makedirs(os.path.join(root, split), exist_ok=True)
        frames = []
        for i, c in enumerate(cams):
            with torch.no_grad():
                out = render(gt, c)
            rgba = torch.cat([out.image.clamp(0, 1), out.alpha[..., None]], -1)
            # store straight (un-premultiplied) colour, as Blender does
            straight = torch.where(out.alpha[..., None] > 0, out.image / out.alpha[..., None].clamp_min(1e-8), 0)
            rgba = torch.cat([straight.clamp(0, 1), out.alpha[..., None]], -1)
            Image.fromarray((rgba.numpy() * 255).round().astype(np.uint8), "RGBA").save(
                os.path.join(root, split, f"r_{i}.png"))
            w2c = np.eye(4)
            w2c[:3, :3] = c.R.double().numpy()
            w2c[:3, 3] = c.t.double().numpy()
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL
            frames.append({"file_path": f"./{split}/r_{i}", "transform_matrix": c2w.tolist()})
        with open(os.path.join(root, f"transforms_{split}.json"), "w") as f:
            json.dump({"camera_angle_x": math.radians(fov_x_deg), "frames": frames}, f)


def test_blender_roundtrip(tmp_path):
    gt = Gaussians.random(30, extent=0.7, sh_degree=0, scale_range=(0.05, 0.2), seed=0)
    train = orbit_cameras(6, 3.0, width=32, height=32, fov_x_deg=40.0, seed=0)
    test = orbit_cameras(2, 3.0, width=32, height=32, fov_x_deg=40.0, seed=1)
    write_blender_dataset(str(tmp_path), {"train": train, "test": test}, gt, 40.0)
    scene = load_blender(str(tmp_path), white_background=True)
    assert len(scene.train) == 6 and len(scene.test) == 2
    for fr, c in zip(scene.train, train):
        assert torch.allclose(fr.camera.R, c.R, atol=1e-5)
        assert torch.allclose(fr.camera.t, c.t, atol=1e-5)
        assert math.isclose(fr.camera.fx, c.fx, rel_tol=1e-9) and fr.camera.cx == 16
        # the loaded image, composited on white, equals a render with a white background
        with torch.no_grad():
            ref = render(gt, c, background=torch.ones(3)).image.clamp(0, 1)
        assert (fr.image_float() - ref).abs().max() < 3 / 255
    assert math.isclose(scene.extent, 1.1 * 3.0 * 1.0, rel_tol=0.25)  # cameras on a radius-3 sphere


def test_blender_downscale(tmp_path):
    gt = Gaussians.random(10, extent=0.5, seed=0)
    cams = orbit_cameras(2, 3.0, width=32, height=24, fov_x_deg=40.0)
    write_blender_dataset(str(tmp_path), {"train": cams, "test": cams[:1]}, gt, 40.0)
    scene = load_blender(str(tmp_path), downscale=2)
    c = scene.train[0].camera
    assert (c.width, c.height) == (16, 12) and scene.train[0].image.shape == (12, 16, 3)
    assert math.isclose(c.fx, cams[0].fx / 2, rel_tol=1e-9)


def test_camera_extent():
    cams = [look_at((x, 0.0, 0.0), (0.0, 0.0, 1.0), width=4, height=4) for x in (-2.0, 0.0, 2.0)]
    assert math.isclose(camera_extent(cams), 2.2, rel_tol=1e-6)


# ------------------------------------------------------------------- COLMAP

def make_colmap_scene(n_images=10, n_points=200, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n_points, 3))
    rgb = rng.integers(0, 256, (n_points, 3)).astype(np.uint8)
    cams = orbit_cameras(n_images, 4.0, width=64, height=48, fov_x_deg=60.0, seed=seed, dtype=torch.float64)
    images = []
    for i, c in enumerate(cams):
        R, t = c.R.numpy(), c.t.numpy()
        pc = pts @ R.T + t
        uv = np.stack([c.fx * pc[:, 0] / pc[:, 2] + c.cx, c.fy * pc[:, 1] / pc[:, 2] + c.cy], -1)
        seen = (uv[:, 0] >= 0) & (uv[:, 0] < 64) & (uv[:, 1] >= 0) & (uv[:, 1] < 48)
        ids = np.where(seen, np.arange(n_points) + 1, -1)
        # quaternion from R (R has positive trace for these cameras? use a robust conversion)
        q = rotmat_to_quat(R)
        images.append(dict(id=i + 1, q=q, t=t, name=f"img_{i:03d}.png", xy=uv, ids=ids))
    cam = dict(id=1, model="PINHOLE", w=64, h=48, params=[cams[0].fx, cams[0].fy, 32.0, 24.0])
    return cam, images, pts, rgb, cams


def rotmat_to_quat(R):
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        return np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def write_colmap_binary(d, cam, images, pts, rgb):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.bin"), "wb") as f:
        mid, n = MODEL_IDS[cam["model"]]
        f.write(struct.pack("<QiiQQ", 1, cam["id"], mid, cam["w"], cam["h"]))
        f.write(struct.pack("<" + "d" * n, *cam["params"]))
    with open(os.path.join(d, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<idddddddi", im["id"], *im["q"], *im["t"], 1))
            f.write(im["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", len(im["ids"])))
            for (x, y), pid in zip(im["xy"], im["ids"]):
                f.write(struct.pack("<ddq", x, y, pid))
    with open(os.path.join(d, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (p, c) in enumerate(zip(pts, rgb)):
            track = [(im["id"], k) for im in images for k, pid in enumerate(im["ids"]) if pid == i + 1]
            f.write(struct.pack("<QdddBBBdQ", i + 1, *p, *c.tolist(), 0.5, len(track)))
            for iid, k in track:
                f.write(struct.pack("<ii", iid, k))


def write_colmap_text(d, cam, images, pts, rgb):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"{cam['id']} {cam['model']} {cam['w']} {cam['h']} {' '.join(repr(float(v)) for v in cam['params'])}\n")
    with open(os.path.join(d, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        for im in images:
            f.write(f"{im['id']} {' '.join(repr(float(v)) for v in im['q'])} {' '.join(repr(float(v)) for v in im['t'])} 1 "
                    f"{im['name']}\n")
            f.write(" ".join(f"{float(x)!r} {float(y)!r} {pid}" for (x, y), pid in zip(im["xy"], im["ids"])) + "\n")
    with open(os.path.join(d, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for i, (p, c) in enumerate(zip(pts, rgb)):
            f.write(f"{i + 1} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} {c[0]} {c[1]} {c[2]} 0.5\n")


def test_colmap_binary_and_text_agree_and_reproject(tmp_path):
    cam, images, pts, rgb, _ = make_colmap_scene()
    write_colmap_binary(str(tmp_path / "bin"), cam, images, pts, rgb)
    write_colmap_text(str(tmp_path / "txt"), cam, images, pts, rgb)
    mb, mt = read_model(str(tmp_path / "bin")), read_model(str(tmp_path / "txt"))
    assert mb.cameras[1].model == mt.cameras[1].model == "PINHOLE"
    assert np.allclose(mb.cameras[1].params, mt.cameras[1].params)
    assert np.allclose(mb.xyz, mt.xyz) and (mb.rgb == mt.rgb).all() and (mb.point_ids == mt.point_ids).all()
    for k in mb.images:
        a, b = mb.images[k], mt.images[k]
        assert a.name == b.name and np.allclose(a.qvec, b.qvec) and np.allclose(a.tvec, b.tvec)
        assert np.allclose(a.xys, b.xys) and (a.point3d_ids == b.point3d_ids).all()
    for m in (mb, mt):
        errs = reprojection_errors(m)
        assert len(errs) > 100 and errs.max() < 1e-6


def test_load_colmap_split_scaling_and_poses(tmp_path):
    cam, images, pts, rgb, cams = make_colmap_scene(n_images=17)
    root = tmp_path / "scene"
    write_colmap_binary(str(root / "sparse" / "0"), cam, images, pts, rgb)
    os.makedirs(root / "images_2")
    for im in images:  # half-resolution images, as in images_2 / images_4 folders
        Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(root / "images_2" / im["name"])
    scene = load_colmap(str(root), images_dir="images_2")
    names = sorted(im["name"] for im in images)
    assert [f.name for f in scene.test] == names[0::8]
    assert len(scene.train) + len(scene.test) == 17
    fr = scene.test[1]  # image index 8
    c = fr.camera
    assert (c.width, c.height) == (32, 24)
    assert math.isclose(c.fx, cams[8].fx / 2, rel_tol=1e-6) and math.isclose(c.cx, 16.0)
    assert torch.allclose(c.R.double(), cams[8].R, atol=1e-6)
    assert torch.allclose(c.t.double(), cams[8].t, atol=1e-6)
    assert scene.points.shape == (200, 3)
