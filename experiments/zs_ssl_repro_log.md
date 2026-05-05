
## 2026-05-05 ZS-SSL first sanity run

- Environment: hfssde_py38
- GPU: RTX 4090
- Data: fastMRI knee multicoil test, file1000015.h5, one selected slice
- Input format: converted to data_hfs_knee_slice.mat
- Memory workaround: center crop to 320 x 320
- Training setting:
  - epochs = 3
  - num_reps = 2
  - nb_unroll_blocks = 2
  - nb_res_blocks = 2
  - CG_Iter = 2
- Result:
  - training completed successfully
  - best.pth saved
  - TrainingLog.mat saved
  - visualization generated: Ref Image / Zero-Filled / ZS-SSL Recon
- Note:
  - This is only a sanity run, not a valid quantitative baseline.

## 2026-05-05 Command-line sanity run completed

- Script: scripts/train_zs_ssl_hfs_knee.py
- Input: data_hfs_knee_slice_crop320.mat
- Output: outputs/zs_ssl_hfs_knee_sanity
- Configuration:
  - epochs = 3
  - stop_training = 3
  - num_reps = 2
  - batch_size = 1
  - nb_unroll_blocks = 2
  - nb_res_blocks = 2
  - CG_Iter = 2
- Result:
  - Epoch 1: train = 0.894747, val = 0.815532
  - Epoch 2: train = 0.882452, val = 0.803100
  - Epoch 3: train = 0.875886, val = 0.801950
  - Training finished successfully in 2.3s
- Status:
  - Command-line ZS-SSL sanity baseline passed.
  - Next step: improve converter to generate crop320 directly and then run longer single-slice training.
