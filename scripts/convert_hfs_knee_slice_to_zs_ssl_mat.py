import os
import glob
import h5py
import numpy as np
import scipy.io as sio


KSPACE_DIR = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/kspace"
MAPS_DIR = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_test//maps"

OUT_MAT = "/mnt/SSD/wsy/projects/SSDU-main/data_hfs_knee_slice.mat"

VOLUME_INDEX = 0
SLICE_INDEX = 10

ACC = 4
ACS = 24


def list_h5_keys(path):
    with h5py.File(path, "r") as f:
        return list(f.keys())


def pick_key(keys, prefer):
    for p in prefer:
        for k in keys:
            if p.lower() in k.lower():
                return k
    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"Cannot infer key from keys={keys}")


def to_complex(arr):
    arr = np.asarray(arr)

    # Already complex
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)

    # Real-imag last channel
    if arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)

    # Real-imag first channel, uncommon
    if arr.shape[0] == 2:
        return (arr[0] + 1j * arr[1]).astype(np.complex64)

    raise RuntimeError(f"Cannot convert to complex. shape={arr.shape}, dtype={arr.dtype}")


def infer_slice_coil_hw_to_hwcoil(x):
    """
    Convert one selected slice to [H, W, C].

    Possible input examples:
      full kspace: [slice, coil, H, W]
      full kspace: [slice, H, W, coil]
      one slice:   [coil, H, W]
      one slice:   [H, W, coil]
    """
    x = np.asarray(x)

    if x.ndim == 4:
        # [slice, coil, H, W]
        if x.shape[1] < 64 and x.shape[2] >= 128 and x.shape[3] >= 128:
            x = x[SLICE_INDEX]
            x = np.transpose(x, (1, 2, 0))  # [H, W, C]
            return x

        # [slice, H, W, coil]
        if x.shape[-1] < 64 and x.shape[1] >= 128 and x.shape[2] >= 128:
            x = x[SLICE_INDEX]
            return x

    if x.ndim == 3:
        # [coil, H, W]
        if x.shape[0] < 64 and x.shape[1] >= 128 and x.shape[2] >= 128:
            return np.transpose(x, (1, 2, 0))

        # [H, W, coil]
        if x.shape[-1] < 64 and x.shape[0] >= 128 and x.shape[1] >= 128:
            return x

    raise RuntimeError(f"Unknown kspace/map layout: shape={x.shape}")


def make_cartesian_mask(shape_hw, acc=4, acs=24):
    """
    Simple equispaced Cartesian mask.
    shape_hw = (H, W)
    Mask samples full ACS center columns and every acc-th outer column.
    """
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.float32)

    center = W // 2
    lo = center - acs // 2
    hi = lo + acs
    mask[:, lo:hi] = 1.0

    # Outer equispaced samples
    mask[:, ::acc] = 1.0

    # Ensure ACS remains fully sampled
    mask[:, lo:hi] = 1.0

    return mask


def main():
    k_files = sorted(glob.glob(os.path.join(KSPACE_DIR, "*.h5")))
    m_files = sorted(glob.glob(os.path.join(MAPS_DIR, "*.h5")))

    print("num kspace files:", len(k_files))
    print("num map files:", len(m_files))

    if not k_files:
        raise RuntimeError(f"No h5 files found in {KSPACE_DIR}")
    if not m_files:
        raise RuntimeError(f"No h5 files found in {MAPS_DIR}")

    k_path = k_files[VOLUME_INDEX]
    m_path = m_files[VOLUME_INDEX]

    print("kspace file:", k_path)
    print("maps file:", m_path)

    k_keys = list_h5_keys(k_path)
    m_keys = list_h5_keys(m_path)

    print("kspace keys:", k_keys)
    print("maps keys:", m_keys)

    k_key = pick_key(k_keys, ["kspace", "k"])
    m_key = pick_key(m_keys, ["maps", "sens", "csm", "s_maps", "sensitivity"])

    print("selected kspace key:", k_key)
    print("selected maps key:", m_key)

    with h5py.File(k_path, "r") as fk:
        k_raw = fk[k_key][()]
    with h5py.File(m_path, "r") as fm:
        m_raw = fm[m_key][()]

    print("raw kspace shape/dtype:", k_raw.shape, k_raw.dtype)
    print("raw maps shape/dtype:", m_raw.shape, m_raw.dtype)

    k_raw = to_complex(k_raw)
    m_raw = to_complex(m_raw)

    kspace = infer_slice_coil_hw_to_hwcoil(k_raw)
    sens_maps = infer_slice_coil_hw_to_hwcoil(m_raw)

    if kspace.shape != sens_maps.shape:
        raise RuntimeError(f"kspace and sens_maps shape mismatch: {kspace.shape} vs {sens_maps.shape}")

    H, W, C = kspace.shape
    mask = make_cartesian_mask((H, W), acc=ACC, acs=ACS)

    print("converted kspace:", kspace.shape, kspace.dtype, "abs max/mean:", np.abs(kspace).max(), np.abs(kspace).mean())
    print("converted sens_maps:", sens_maps.shape, sens_maps.dtype, "abs max/mean:", np.abs(sens_maps).max(), np.abs(sens_maps).mean())
    print("mask:", mask.shape, mask.dtype, "sampled ratio:", mask.mean())

    os.makedirs(os.path.dirname(OUT_MAT), exist_ok=True)
    sio.savemat(
        OUT_MAT,
        {
            "kspace": kspace.astype(np.complex64),
            "sens_maps": sens_maps.astype(np.complex64),
            "mask": mask.astype(np.float32),
        },
    )

    print("saved:", OUT_MAT)


if __name__ == "__main__":
    main()
