"""Download and compare FoCUS scattering coefficients across STL, FOSCAT, and NumPy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from foscat.numpy_scat2d import compute_scattering


def _download_npy(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _load_image(url: str, index: int, cache_dir: Path) -> np.ndarray:
    fname = cache_dir / "cloud_fields.npy"
    _download_npy(url, fname)
    data = np.load(fname)
    if index < 0 or index >= data.shape[0]:
        raise IndexError(f"index {index} out of bounds for array with shape {data.shape}")
    return data[index].astype(np.float32)


def _maybe_load_stl(path: Optional[str]):
    if path is None:
        return None, None

    sys.path.append(path)
    try:
        from STL_2D_Kernel_Torch import STL_2D_Kernel_Torch as STLDataClass
        from ST_Operator import ST_Operator as SO
    except Exception as exc:  # pragma: no cover - runtime import guard
        print(f"Could not import STL from {path}: {exc}")
        return None, None
    return STLDataClass, SO


def _compute_stl(image: np.ndarray, stl_cls, st_operator):
    dc = stl_cls(image)
    st_op = st_operator(dc)
    return st_op.apply(dc)


def _compute_foscat(image: np.ndarray, jmax: int = 6, norient: int = 4):
    import foscat.scat_cov2D as sc

    scat_op = sc.funct(NORIENT=norient)
    coeffs, _ = scat_op.eval(image, calc_var=True, Jmax=jmax)
    return coeffs


def _plot_semilogy(ax, values_a, values_b, title: str, label_a: str, label_b: str):
    ax.plot(values_a.flatten(), color="blue")
    ax.plot(values_b.flatten(), color="orange")
    ax.legend([label_a, label_b])
    ax.set_title(title)
    ax.semilogy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="https://github.com/jmdelouis/FOSCAT/raw/main/src/tests/cloud_fields.npy",
        help="URL to the numpy image array (expects a stacked image cube)",
    )
    parser.add_argument("--index", type=int, default=1, help="Index of the image to load")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./data_cache"),
        help="Directory to cache the downloaded .npy file",
    )
    parser.add_argument(
        "--stl-path",
        default="/Users/pcampeti/STL_dev_old/STL_main",
        help="Path containing STL_2D_Kernel_Torch and ST_Operator modules",
    )
    parser.add_argument(
        "--kernelsz",
        type=int,
        default=5,
        help="Kernel size for NumPy scattering (3, 5, or 9)",
    )
    args = parser.parse_args()

    image = _load_image(args.url, args.index, args.cache_dir)

    # STL reference
    stl_cls, st_operator = _maybe_load_stl(args.stl_path)
    stl_result = _compute_stl(image, stl_cls, st_operator) if stl_cls else None

    # FOSCAT reference
    foscat_result = _compute_foscat(image)

    # NumPy scattering
    numpy_result = compute_scattering(image, KERNELSZ=args.kernelsz)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    if stl_result is not None:
        _plot_semilogy(
            axes[0, 0],
            stl_result.S1[0, 0].cpu().numpy(),
            foscat_result.S1[0, 0].cpu().numpy(),
            "S1",
            "STL2D Kernel",
            "FOSCAT",
        )
        _plot_semilogy(
            axes[0, 1],
            stl_result.S2[0, 0].cpu().numpy(),
            foscat_result.S2[0, 0].cpu().numpy(),
            "S2",
            "STL2D Kernel",
            "FOSCAT",
        )
        _plot_semilogy(
            axes[1, 0],
            stl_result.S3[0, 0].cpu().numpy(),
            foscat_result.S3[0, 0].cpu().numpy(),
            "S3",
            "STL2D Kernel",
            "FOSCAT",
        )
        _plot_semilogy(
            axes[1, 1],
            stl_result.S4[0, 0].cpu().numpy(),
            foscat_result.S4[0, 0].cpu().numpy(),
            "S4",
            "STL2D Kernel",
            "FOSCAT",
        )
    else:
        axes[0, 0].text(0.5, 0.5, "STL not available", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.suptitle("STL vs FOSCAT scattering coefficients")
    fig.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    axes2[0].plot(numpy_result.S1[0].flatten(), color="green")
    axes2[0].plot(foscat_result.S1[0, 0].cpu().numpy().flatten(), color="orange")
    axes2[0].legend(["NumPy", "FOSCAT"])
    axes2[0].set_title("NumPy vs FOSCAT S1")
    axes2[0].semilogy()

    axes2[1].plot(np.abs(numpy_result.S2[0]).flatten(), color="green")
    axes2[1].plot(np.abs(foscat_result.S2[0, 0].cpu().numpy()).flatten(), color="orange")
    axes2[1].legend(["NumPy", "FOSCAT"])
    axes2[1].set_title("NumPy vs FOSCAT |S2|")
    axes2[1].semilogy()

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
