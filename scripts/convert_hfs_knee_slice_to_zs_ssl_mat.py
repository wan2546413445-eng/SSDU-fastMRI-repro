import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


def make_cartesian_mask(height: int, width: int, acc: int, acs: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, ::acc] = 1.0
    c0 = max(0, (width - acs) // 2)
    c1 = min(width, c0 + acs)
    mask[:, c0:c1] = 1.0
    return mask


def center_crop_hw(x: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = x.shape[:2]
    if crop_size <= 0:
        return x
    if crop_size > h or crop_size > w:
        raise ValueError(f'crop_size={crop_size} exceeds input spatial shape {(h, w)}')
    hs = (h - crop_size) // 2
    ws = (w - crop_size) // 2
    if x.ndim == 3:
        return x[hs:hs + crop_size, ws:ws + crop_size, :]
    return x[hs:hs + crop_size, ws:ws + crop_size]


def main():
    parser = argparse.ArgumentParser(description='Convert one HFS/fastMRI knee slice into ZS-SSL data.mat format')
    parser.add_argument('--kspace_dir', type=str, default='/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/kspace')
    parser.add_argument('--maps_dir', type=str, default='/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/maps')
    parser.add_argument('--volume_index', type=int, default=0)
    parser.add_argument('--slice_index', type=int, default=10)
    parser.add_argument('--acc', type=int, default=4)
    parser.add_argument('--acs', type=int, default=24)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--out', type=str, default='/mnt/SSD/wsy/projects/SSDU-main/data_hfs_knee_slice_crop320.mat')
    args = parser.parse_args()

    k_files = sorted(Path(args.kspace_dir).glob('*.h5'))
    m_files = sorted(Path(args.maps_dir).glob('*.h5'))
    print(f'[INFO] kspace file count: {len(k_files)}')
    print(f'[INFO] maps file count:   {len(m_files)}')
    if len(k_files) == 0 or len(m_files) == 0:
        raise FileNotFoundError('No .h5 files found in kspace_dir or maps_dir.')

    k_path = k_files[args.volume_index]
    m_path = m_files[args.volume_index]
    print(f'[INFO] selected kspace file: {k_path}')
    print(f'[INFO] selected maps file:   {m_path}')

    with h5py.File(k_path, 'r') as fk, h5py.File(m_path, 'r') as fm:
        k_key = next(iter(fk.keys()))
        m_key = next(iter(fm.keys()))
        print(f'[INFO] raw kspace dataset key={k_key}, shape={fk[k_key].shape}, dtype={fk[k_key].dtype}')
        print(f'[INFO] raw maps dataset key={m_key}, shape={fm[m_key].shape}, dtype={fm[m_key].dtype}')
        k_slice = fk[k_key][args.slice_index]
        m_slice = fm[m_key][args.slice_index]

    print(f'[INFO] selected kspace slice shape={k_slice.shape}, dtype={k_slice.dtype}')
    print(f'[INFO] selected maps slice shape={m_slice.shape}, dtype={m_slice.dtype}')

    kspace = np.transpose(k_slice, (1, 2, 0)).astype(np.complex64)
    sens_maps = np.transpose(m_slice, (1, 2, 0)).astype(np.complex64)

    if args.crop_size > 0:
        kspace = center_crop_hw(kspace, args.crop_size)
        sens_maps = center_crop_hw(sens_maps, args.crop_size)

    h, w, _ = kspace.shape
    mask = make_cartesian_mask(h, w, args.acc, args.acs).astype(np.float32)

    print(f'[INFO] converted kspace shape={kspace.shape}, dtype={kspace.dtype}')
    print(f'[INFO] converted sens_maps shape={sens_maps.shape}, dtype={sens_maps.dtype}')
    print(f'[INFO] converted mask shape={mask.shape}, dtype={mask.dtype}')
    print(f'[INFO] |kspace| max={np.abs(kspace).max():.6g}, mean={np.abs(kspace).mean():.6g}')
    print(f'[INFO] |sens_maps| max={np.abs(sens_maps).max():.6g}, mean={np.abs(sens_maps).mean():.6g}')
    print(f'[INFO] mask sampled ratio={mask.mean():.6f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), {
        'kspace': kspace,
        'sens_maps': sens_maps,
        'mask': mask,
        'volume_index': np.int32(args.volume_index),
        'slice_index': np.int32(args.slice_index),
        'acc': np.int32(args.acc),
        'acs': np.int32(args.acs),
        'crop_size': np.int32(args.crop_size),
    })
    print(f'[INFO] output path: {out_path}')


if __name__ == '__main__':
    main()
