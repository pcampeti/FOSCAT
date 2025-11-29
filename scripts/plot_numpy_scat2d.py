"""Plot FoCUS scattering coefficients with the NumPy-only implementation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from foscat.numpy_scat2d import compute_scattering


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA images to a single luminance channel."""
    if img.ndim == 2:
        return img.astype(np.float32)

    if img.ndim == 3 and img.shape[2] in (3, 4):
        rgb = img[..., :3].astype(np.float32)
        weights = np.array([0.2989, 0.587, 0.114], dtype=np.float32)
        return (rgb * weights).sum(axis=-1)

    raise ValueError("Expected grayscale or RGB/RGBA image")


def _format_s2_labels(j1: np.ndarray, j2: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    labels = [f"{i}->{j}" for i, j in zip(j1, j2)]
    ticks = np.arange(len(labels))
    return ticks, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a 2D image file")
    parser.add_argument(
        "--kernelsz", type=int, default=5, help="FoCUS kernel size (3, 5, or 9)"
    )
    args = parser.parse_args()

    img = _to_grayscale(imageio.imread(args.image))
    coeffs = compute_scattering(img, KERNELSZ=args.kernelsz)

    s1 = coeffs.S1[0]  # [jmax, NORIENT]
    s2 = np.abs(coeffs.S2[0])  # [j_pairs, NORIENT]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    im1 = axes[1].imshow(s1, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("S1 coefficients")
    axes[1].set_xlabel("Orientation")
    axes[1].set_ylabel("Scale j1")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    ticks, labels = _format_s2_labels(coeffs.j1, coeffs.j2)
    im2 = axes[2].imshow(s2, aspect="auto", origin="lower", cmap="magma")
    axes[2].set_title("S2 coefficients |S2|")
    axes[2].set_xlabel("Orientation")
    axes[2].set_ylabel("(j1 -> j2) pair")
    axes[2].set_yticks(ticks)
    axes[2].set_yticklabels(labels)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
