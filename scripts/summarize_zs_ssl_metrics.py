import argparse
import csv
from pathlib import Path
import numpy as np
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def metrics(ref, img):
    nmse = np.linalg.norm(img - ref) ** 2 / (np.linalg.norm(ref) ** 2 + 1e-12)
    psnr = peak_signal_noise_ratio(ref, img, data_range=1.0)
    ssim = structural_similarity(ref, img, data_range=1.0)
    return nmse, psnr, ssim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--csv', required=True)
    args = p.parse_args()

    rows = []
    for f in args.inputs:
        d = sio.loadmat(f)
        ref = np.abs(np.asarray(d['ref']).squeeze())
        zf = np.abs(np.asarray(d['zf']).squeeze())
        recon = np.abs(np.asarray(d['recon']).squeeze())
        scale = np.max(np.abs(ref)) + 1e-12
        ref_n, zf_n, recon_n = ref / scale, zf / scale, recon / scale
        zf_nmse, zf_psnr, zf_ssim = metrics(ref_n, zf_n)
        rc_nmse, rc_psnr, rc_ssim = metrics(ref_n, recon_n)
        rows.append({
            'file': f,
            'zf_nmse': zf_nmse, 'zf_psnr': zf_psnr, 'zf_ssim': zf_ssim,
            'recon_nmse': rc_nmse, 'recon_psnr': rc_psnr, 'recon_ssim': rc_ssim,
        })

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['file', 'zf_nmse', 'zf_psnr', 'zf_ssim', 'recon_nmse', 'recon_psnr', 'recon_ssim']
    with open(out, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
