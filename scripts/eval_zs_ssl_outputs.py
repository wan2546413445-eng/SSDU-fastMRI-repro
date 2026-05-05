import argparse
import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim


def nmse(ref, pred):
    return np.linalg.norm(ref - pred) ** 2 / (np.linalg.norm(ref) ** 2 + 1e-12)


def psnr(ref, pred, data_range=1.0):
    mse = np.mean((ref - pred) ** 2)
    return 20 * np.log10(data_range / (np.sqrt(mse) + 1e-12))


def norm01(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.min()
    return x / (x.max() + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, required=True)
    args = parser.parse_args()

    data = sio.loadmat(args.mat)
    ref = np.asarray(data["ref"]).squeeze()
    zf = np.asarray(data["zf"]).squeeze()
    recon = np.asarray(data["recon"]).squeeze()

    ref_n = norm01(ref)
    zf_n = norm01(zf)
    recon_n = norm01(recon)

    print("Shapes:")
    print("  ref:", ref.shape, "zf:", zf.shape, "recon:", recon.shape)

    print("\nMetrics against ref image after [0,1] normalization:")
    print(f"  ZF     PSNR={psnr(ref_n, zf_n):.4f}  SSIM={ssim(ref_n, zf_n, data_range=1.0):.4f}  NMSE={nmse(ref_n, zf_n):.6f}")
    print(f"  Recon  PSNR={psnr(ref_n, recon_n):.4f}  SSIM={ssim(ref_n, recon_n, data_range=1.0):.4f}  NMSE={nmse(ref_n, recon_n):.6f}")


if __name__ == "__main__":
    main()
