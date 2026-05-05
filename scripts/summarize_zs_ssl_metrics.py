import argparse
import csv
import glob
import os
import re

import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim


def norm01(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.min()
    return x / (x.max() + 1e-12)


def psnr(ref, pred, data_range=1.0):
    mse = np.mean((ref - pred) ** 2)
    return 20 * np.log10(data_range / (np.sqrt(mse) + 1e-12))


def nmse(ref, pred):
    return np.linalg.norm(ref - pred) ** 2 / (np.linalg.norm(ref) ** 2 + 1e-12)


def parse_slice_id(path):
    m = re.search(r"slice(\d+)", path)
    return int(m.group(1)) if m else -1


def compute_one(mat_path):
    data = sio.loadmat(mat_path)
    ref = norm01(np.asarray(data["ref"]).squeeze())
    zf = norm01(np.asarray(data["zf"]).squeeze())
    recon = norm01(np.asarray(data["recon"]).squeeze())

    zf_psnr = psnr(ref, zf)
    rec_psnr = psnr(ref, recon)
    zf_ssim = ssim(ref, zf, data_range=1.0)
    rec_ssim = ssim(ref, recon, data_range=1.0)
    zf_nmse = nmse(ref, zf)
    rec_nmse = nmse(ref, recon)

    return {
        "mat_path": mat_path,
        "slice": parse_slice_id(mat_path),
        "zf_psnr": zf_psnr,
        "recon_psnr": rec_psnr,
        "delta_psnr": rec_psnr - zf_psnr,
        "zf_ssim": zf_ssim,
        "recon_ssim": rec_ssim,
        "delta_ssim": rec_ssim - zf_ssim,
        "zf_nmse": zf_nmse,
        "recon_nmse": rec_nmse,
        "delta_nmse": rec_nmse - zf_nmse,
        "nmse_reduction_pct": 100.0 * (zf_nmse - rec_nmse) / (zf_nmse + 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, required=True, help="Glob pattern for recon_outputs.mat files.")
    parser.add_argument("--csv", type=str, default="outputs/zs_ssl_metrics_summary.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise RuntimeError(f"No files matched pattern: {args.glob}")

    rows = [compute_one(p) for p in files]
    rows = sorted(rows, key=lambda r: r["slice"])

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {args.csv}")
    print("\nPer-slice metrics:")
    for r in rows:
        print(
            f"slice={r['slice']:>3} | "
            f"ZF/Recon PSNR={r['zf_psnr']:.4f}/{r['recon_psnr']:.4f} "
            f"Δ={r['delta_psnr']:+.4f} | "
            f"ZF/Recon SSIM={r['zf_ssim']:.4f}/{r['recon_ssim']:.4f} "
            f"Δ={r['delta_ssim']:+.4f} | "
            f"ZF/Recon NMSE={r['zf_nmse']:.6f}/{r['recon_nmse']:.6f} "
            f"reduction={r['nmse_reduction_pct']:.2f}%"
        )

    print("\nAverages:")
    for key in ["zf_psnr", "recon_psnr", "delta_psnr", "zf_ssim", "recon_ssim", "delta_ssim", "zf_nmse", "recon_nmse", "delta_nmse", "nmse_reduction_pct"]:
        print(f"{key}: {np.mean([r[key] for r in rows]):.6f}")


if __name__ == "__main__":
    main()
