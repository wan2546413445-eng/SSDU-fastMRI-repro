import argparse
import numpy as np
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mat', required=True)
    args = p.parse_args()
    d = sio.loadmat(args.mat)
    ref = np.abs(np.asarray(d['ref']).squeeze())
    zf = np.abs(np.asarray(d['zf']).squeeze())
    recon = np.abs(np.asarray(d['recon']).squeeze())

    scale = np.max(np.abs(ref)) + 1e-12
    ref_n, zf_n, recon_n = ref / scale, zf / scale, recon / scale

    for name, x in [('zf', zf_n), ('recon', recon_n)]:
        nmse = np.linalg.norm(x - ref_n) ** 2 / (np.linalg.norm(ref_n) ** 2 + 1e-12)
        psnr = peak_signal_noise_ratio(ref_n, x, data_range=1.0)
        ssim = structural_similarity(ref_n, x, data_range=1.0)
        print(f'{name}: NMSE={nmse:.6e} PSNR={psnr:.4f} SSIM={ssim:.6f}')


if __name__ == '__main__':
    main()
