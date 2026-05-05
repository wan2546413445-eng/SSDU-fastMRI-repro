
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
