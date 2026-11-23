"""Dataset loaders: NeRF-synthetic (Blender) and COLMAP scenes.

Both produce a `Scene`: train / test frames (camera + image), the background
colour, the scene extent used to scale learning rates and density thresholds,
and optional SfM points for initialisation.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .camera import Camera
from .colmap import read_model


@dataclass
class Frame:
    camera: Camera
    image: torch.Tensor      # (H, W, 3) uint8 (kept compact; converted per iteration)
    name: str

    def image_float(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return self.image.to(device=device).to(dtype) / 255.0


@dataclass
class Scene:
    train: list[Frame]
    test: list[Frame]
    background: torch.Tensor          # (3,)
    extent: float                     # radius of the camera cloud x 1.1 (the paper's "cameras_extent")
    points: np.ndarray | None = None  # (P, 3) SfM points
    colors: np.ndarray | None = None  # (P, 3) in [0, 1]


def camera_extent(cams: list[Camera]) -> float:
    """1.1 x the largest distance of a camera centre from their mean (getNerfppNorm in 3DGS)."""
    centers = torch.stack([c.center.double() for c in cams])
    return 1.1 * float((centers - centers.mean(0)).norm(dim=-1).max())


def _to_uint8_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(arr, copy=True)).to(torch.uint8)


# --------------------------------------------------------------------- Blender

def blender_camera(transform_matrix, camera_angle_x: float, width: int, height: int) -> Camera:
    """NeRF-synthetic stores camera->world in OpenGL axes (x right, y up, z backward).
    Flip y and z to get OpenCV axes, then invert to world->camera."""
    c2w = np.array(transform_matrix, dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    fx = 0.5 * width / math.tan(0.5 * camera_angle_x)
    return Camera(torch.tensor(w2c[:3, :3], dtype=torch.float32), torch.tensor(w2c[:3, 3], dtype=torch.float32),
                  fx, fx, width / 2, height / 2, width, height)


def _load_blender_split(root, split, downscale, background):
    with open(os.path.join(root, f"transforms_{split}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    frames = []
    for fr in meta["frames"]:
        path = os.path.join(root, fr["file_path"] + ("" if fr["file_path"].endswith(".png") else ".png"))
        img = Image.open(path).convert("RGBA")
        if downscale != 1:
            img = img.resize((round(img.width / downscale), round(img.height / downscale)), Image.BICUBIC)
        rgba = np.asarray(img).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:] + background[None, None, :] * (1 - rgba[..., 3:])
        cam = blender_camera(fr["transform_matrix"], meta["camera_angle_x"], img.width, img.height)
        frames.append(Frame(cam, _to_uint8_tensor(np.round(rgb * 255)), os.path.basename(path)))
    return frames


def load_blender(root: str, *, downscale: float = 1, white_background: bool = True,
                 test_split: str = "test") -> Scene:
    bg = np.array([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], dtype=np.float32)
    train = _load_blender_split(root, "train", downscale, bg)
    test = _load_blender_split(root, test_split, downscale, bg)
    return Scene(train, test, torch.tensor(bg), camera_extent([f.camera for f in train]))


def random_init_points(n: int, half_size: float = 1.3, seed: int = 0):
    """Random points in a cube (the paper's init for NeRF-synthetic: 100k in [-1.3, 1.3]^3)."""
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3)) * 2 - 1) * half_size
    cols = rng.random((n, 3))
    return pts, cols


# ---------------------------------------------------------------------- COLMAP

def load_colmap(root: str, *, images_dir: str = "images", llffhold: int = 8, max_frames: int | None = None) -> Scene:
    """COLMAP scene: root/sparse/0 + root/<images_dir>. Every `llffhold`-th image (sorted by
    name, starting at 0) is held out for testing, the standard Mip-NeRF 360 protocol. Images in a
    downsampled folder (images_4 ...) keep the poses; intrinsics are scaled to the image size."""
    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isdir(sparse):
        sparse = os.path.join(root, "sparse")
    model = read_model(sparse)
    images = sorted(model.images.values(), key=lambda im: im.name)
    if max_frames:
        images = images[:max_frames]
    train, test = [], []
    for k, im in enumerate(images):
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = cam.pinhole()
        path = os.path.join(root, images_dir, im.name)
        if not os.path.exists(path):  # downsampled folders sometimes change the extension
            stem = os.path.splitext(path)[0]
            path = next(stem + e for e in (".png", ".jpg", ".JPG", ".jpeg") if os.path.exists(stem + e))
        img = Image.open(path).convert("RGB")
        sx, sy = img.width / cam.width, img.height / cam.height
        c = Camera(torch.tensor(im.rotmat(), dtype=torch.float32), torch.tensor(im.tvec, dtype=torch.float32),
                   fx * sx, fy * sy, cx * sx, cy * sy, img.width, img.height)
        fr = Frame(c, _to_uint8_tensor(np.asarray(img)), im.name)
        (test if k % llffhold == 0 else train).append(fr)
    return Scene(train, test, torch.zeros(3), camera_extent([f.camera for f in train]),
                 model.xyz, model.rgb.astype(np.float64) / 255.0)
