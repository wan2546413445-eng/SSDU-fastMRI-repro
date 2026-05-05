import argparse
import glob
import os

import h5py
import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one HFS/fastMRI knee multicoil slice to ZS-SSL data.mat format."
    )
    parser.add_argument(
        "--kspace_dir",
        type=str,
        default="/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/kspace",
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/maps",
    )
    parser.add_argument("--volume_index", type=int, default=0)
    parser.add_argument("--slice_index", type=int, default=10)
    parser.add_argument("--acc", type=int, default=4)
    parser.add_argument("--acs", type=int, default=24)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument(
        "--out",
        type=str,
        default="/mnt/SSD/wsy/projects/SSDU-main/data_hfs_knee_slice_crop320.mat",
    )
    return parser.parse_args()


def pick_dataset_key(h5_file, exact_names, fallback_contains=None):
    keys = list(h5_file.keys())
    fallback_contains = fallback_contains or []

    # 1. Exact match first. This avoids accidentally selecting ismrmrd_header.
    for name in exact_names:
        if name in keys:
            return name

    # 2. Fallback: substring match, but only for array-like datasets with ndim >= 3.
    for token in fallback_contains:
        for key in keys:
            obj = h5_file[key]
            if token.lower() in key.lower() and hasattr(obj, "shape") and len(obj.shape) >= 3:
                return key

    raise RuntimeError(f"Cannot infer dataset key from keys={keys}")


def to_complex(arr):
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)

    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)

    if arr.ndim >= 1 and arr.shape[0] == 2:
        return (arr[0] + 1j * arr[1]).astype(np.complex64)

    raise RuntimeError(f"Cannot convert to complex: shape={arr.shape}, dtype={arr.dtype}")


def one_slice_to_hwcoil(x):
    """
    Convert one selected slice to [H, W, coil].

    Expected common layouts after slice selection:
      [coil, H, W] -> [H, W, coil]
      [H, W, coil] -> [H, W, coil]
    """
    x = np.squeeze(np.asarray(x))

    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D slice after squeeze, got shape={x.shape}")

    # [coil, H, W]
    if x.shape[0] < 64 and x.shape[1] >= 128 and x.shape[2] >= 128:
        return np.transpose(x, (1, 2, 0))

    # [H, W, coil]
    if x.shape[-1] < 64 and x.shape[0] >= 128 and x.shape[1] >= 128:
        return x

    raise RuntimeError(f"Unknown slice layout: shape={x.shape}")


def make_cartesian_mask(shape_hw, acc, acs):
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.float32)

    center = W // 2
    lo = center - acs // 2
    hi = lo + acs

    mask[:, ::acc] = 1.0
    mask[:, lo:hi] = 1.0

    return mask


def center_crop_hw(x, crop_size):
    if crop_size is None or crop_size <= 0:
        return x

    H, W = x.shape[:2]
    if crop_size > H or crop_size > W:
        raise RuntimeError(f"crop_size={crop_size} exceeds input shape={x.shape}")

    h0 = H // 2 - crop_size // 2
    w0 = W // 2 - crop_size // 2

    if x.ndim == 2:
        return x[h0:h0 + crop_size, w0:w0 + crop_size]

    if x.ndim == 3:
        return x[h0:h0 + crop_size, w0:w0 + crop_size, :]

    raise RuntimeError(f"Unsupported crop input shape={x.shape}")


def main():
    args = parse_args()

    k_files = sorted(glob.glob(os.path.join(args.kspace_dir, "*.h5")))
    m_files = sorted(glob.glob(os.path.join(args.maps_dir, "*.h5")))

    print(f"[INFO] kspace file count: {len(k_files)}")
    print(f"[INFO] maps file count:   {len(m_files)}")

    if not k_files:
        raise RuntimeError(f"No kspace h5 files found in {args.kspace_dir}")
    if not m_files:
        raise RuntimeError(f"No map h5 files found in {args.maps_dir}")

    if args.volume_index < 0 or args.volume_index >= len(k_files):
        raise RuntimeError(f"volume_index={args.volume_index} out of range for {len(k_files)} kspace files")
    if args.volume_index >= len(m_files):
        raise RuntimeError(f"volume_index={args.volume_index} out of range for {len(m_files)} map files")

    k_path = k_files[args.volume_index]
    m_path = m_files[args.volume_index]

    print(f"[INFO] selected kspace file: {k_path}")
    print(f"[INFO] selected maps file:   {m_path}")

    with h5py.File(k_path, "r") as fk, h5py.File(m_path, "r") as fm:
        print(f"[INFO] kspace keys: {list(fk.keys())}")
        print(f"[INFO] maps keys:   {list(fm.keys())}")

        k_key = pick_dataset_key(fk, exact_names=["kspace"], fallback_contains=["kspace"])
        m_key = pick_dataset_key(fm, exact_names=["s_maps", "maps"], fallback_contains=["sens", "map", "csm"])

        k_ds = fk[k_key]
        m_ds = fm[m_key]

        print(f"[INFO] raw kspace dataset key={k_key}, shape={k_ds.shape}, dtype={k_ds.dtype}")
        print(f"[INFO] raw maps dataset key={m_key}, shape={m_ds.shape}, dtype={m_ds.dtype}")

        if len(k_ds.shape) < 4:
            raise RuntimeError(f"Selected kspace key={k_key} is not a volume dataset: shape={k_ds.shape}")
        if len(m_ds.shape) < 4:
            raise RuntimeError(f"Selected maps key={m_key} is not a volume dataset: shape={m_ds.shape}")

        if args.slice_index < 0 or args.slice_index >= k_ds.shape[0]:
            raise RuntimeError(f"slice_index={args.slice_index} out of range for kspace shape={k_ds.shape}")
        if args.slice_index >= m_ds.shape[0]:
            raise RuntimeError(f"slice_index={args.slice_index} out of range for maps shape={m_ds.shape}")

        k_slice = k_ds[args.slice_index]
        m_slice = m_ds[args.slice_index]

    print(f"[INFO] selected kspace slice shape={k_slice.shape}, dtype={k_slice.dtype}")
    print(f"[INFO] selected maps slice shape={m_slice.shape}, dtype={m_slice.dtype}")

    k_slice = to_complex(k_slice)
    m_slice = to_complex(m_slice)

    kspace = one_slice_to_hwcoil(k_slice)
    sens_maps = one_slice_to_hwcoil(m_slice)

    if kspace.shape != sens_maps.shape:
        raise RuntimeError(f"kspace/sens_maps mismatch: {kspace.shape} vs {sens_maps.shape}")

    H, W, C = kspace.shape
    mask = make_cartesian_mask((H, W), acc=args.acc, acs=args.acs)

    print(f"[INFO] converted kspace shape={kspace.shape}, dtype={kspace.dtype}, abs max/mean={np.abs(kspace).max():.6g}/{np.abs(kspace).mean():.6g}")
    print(f"[INFO] converted maps   shape={sens_maps.shape}, dtype={sens_maps.dtype}, abs max/mean={np.abs(sens_maps).max():.6g}/{np.abs(sens_maps).mean():.6g}")
    print(f"[INFO] mask shape={mask.shape}, dtype={mask.dtype}, sampled ratio={mask.mean():.6f}")

    if args.crop_size and args.crop_size > 0:
        kspace = center_crop_hw(kspace, args.crop_size)
        sens_maps = center_crop_hw(sens_maps, args.crop_size)
        mask = center_crop_hw(mask, args.crop_size)
        print(f"[INFO] after crop_size={args.crop_size}:")
        print(f"       kspace={kspace.shape}, sens_maps={sens_maps.shape}, mask={mask.shape}, sampled ratio={mask.mean():.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sio.savemat(
        args.out,
        {
            "kspace": kspace.astype(np.complex64),
            "sens_maps": sens_maps.astype(np.complex64),
            "mask": mask.astype(np.float32),
        },
    )

    print(f"[INFO] saved: {args.out}")


if __name__ == "__main__":
    main()
