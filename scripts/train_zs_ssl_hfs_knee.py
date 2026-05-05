import os
import time

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from models import UnrollNet, parser_ops, utils
from models.modules import Dataset, Dataset_Inference, MixL1L2Loss, test, train, validation


def build_args():
    parser = parser_ops.get_parser()
    parser.add_argument('--data_path', type=str, default='/mnt/SSD/wsy/projects/SSDU-main/data_hfs_knee_slice_crop320.mat')
    parser.add_argument('--output_dir', type=str, default='outputs/zs_ssl_hfs_knee')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cg_iter', type=int, default=2)
    args = parser.parse_args()

    # Required sanity defaults while keeping existing parser fields.
    args.data_dir = args.data_path
    args.batchSize = args.batch_size
    args.CG_Iter = args.cg_iter
    if args.epochs == parser.get_default('epochs'):
        args.epochs = 3
    if args.stop_training == parser.get_default('stop_training'):
        args.stop_training = 3
    if args.num_reps == parser.get_default('num_reps'):
        args.num_reps = 2
    if args.nb_unroll_blocks == parser.get_default('nb_unroll_blocks'):
        args.nb_unroll_blocks = 2
    if args.nb_res_blocks == parser.get_default('nb_res_blocks'):
        args.nb_res_blocks = 2
    return args


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat(args.data_path)
    kspace_train, sens_maps, original_mask = data['kspace'], data['sens_maps'], data['mask']
    args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB = kspace_train.shape
    kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

    cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train, original_mask, rho=args.rho_val)
    remainder_mask, cv_val_mask = np.copy(cv_trn_mask), np.copy(np.complex64(cv_val_mask))

    nw_input_val = utils.sense1(
        kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)), sens_maps
    )[np.newaxis]
    ref_kspace_val = (kspace_train * np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)))[np.newaxis]

    nw_input_trn = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    trn_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    for jj in range(args.num_reps):
        trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train, remainder_mask, rho=args.rho_train)
        sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        nw_input_trn[jj, ...] = utils.sense1(sub_kspace, sens_maps)

    if args.data_opt == 'Coronal_PD':
        trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))
        trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

    sens_maps = np.tile(sens_maps[np.newaxis], (args.num_reps, 1, 1, 1))
    sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
    ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input_trn = utils.complex2real(nw_input_trn)
    ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))
    nw_input_val = utils.complex2real(nw_input_val)

    train_loader = DataLoader(Dataset(nw_input_trn, trn_mask, loss_mask, sens_maps, ref_kspace),
                              batch_size=args.batchSize, shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(nw_input_val, cv_trn_mask[np.newaxis], cv_val_mask[np.newaxis], sens_maps[0][np.newaxis], ref_kspace_val),
                            batch_size=args.batchSize, shuffle=False, num_workers=0)

    model = UnrollNet.UnrolledNet(args, device=device).to(device)
    loss_fn = MixL1L2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_train_loss, total_val_loss = [], []
    valid_loss_min, ep, val_loss_tracker = np.inf, 0, 0
    start_time = time.time()

    while ep < args.epochs and val_loss_tracker < args.stop_training:
        trn_loss, _ = train(train_loader, model, loss_fn, optimizer, device=device)
        val_loss = validation(val_loader, model, loss_fn, device=device)
        total_train_loss.append(trn_loss)
        total_val_loss.append(val_loss)

        checkpoint = {'epoch': ep, 'valid_loss_min': val_loss, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}
        if val_loss <= valid_loss_min:
            valid_loss_min = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            val_loss_tracker = 0
        else:
            val_loss_tracker += 1
        print(f'Epoch:{ep + 1}/{args.epochs} train={trn_loss:.6f} val={val_loss:.6f}')
        ep += 1

    print(f'Training finished in {time.time() - start_time:.1f}s')

    test_mask = np.complex64(original_mask)
    nw_input_inference = utils.sense1(kspace_train * np.tile(test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)),
                                      np.transpose(sens_maps[0], (1, 2, 0)))
    ref_image = utils.sense1(kspace_train, np.transpose(sens_maps[0], (1, 2, 0)))
    if args.data_opt == 'Coronal_PD':
        test_mask[:, 0:17] = np.ones((args.nrow_GLOB, 17))
        test_mask[:, 352:args.ncol_GLOB] = np.ones((args.nrow_GLOB, 16))

    test_loader = DataLoader(Dataset_Inference(utils.complex2real(nw_input_inference[np.newaxis]),
                                               test_mask[np.newaxis], test_mask[np.newaxis], sens_maps[0][np.newaxis]),
                             batch_size=args.batchSize, shuffle=False, num_workers=0)

    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state'])
    zs_ssl_recon = utils.real2complex(test(test_loader, model, device).to('cpu').numpy())

    factor = np.max(np.abs(ref_image[:])) if args.data_opt == 'Coronal_PD' else 1
    ref_image_abs = np.abs(ref_image) / factor
    zf_image_abs = np.abs(nw_input_inference) / factor
    recon_abs = np.abs(zs_ssl_recon) / factor

    sio.savemat(os.path.join(args.output_dir, 'recon_outputs.mat'), {'recon': recon_abs, 'zf': zf_image_abs, 'ref': ref_image_abs})
    sio.savemat(os.path.join(args.output_dir, 'TrainingLog.mat'), {'total_train_loss': np.asarray(total_train_loss), 'total_val_loss': np.asarray(total_val_loss)})

    plt.figure(figsize=(16, 6))
    vmax = 0.6 * np.max(ref_image_abs[:])
    plt.subplot(1, 3, 1); plt.imshow(ref_image_abs, cmap='gray', vmax=vmax); plt.title('Ref Image'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(zf_image_abs, cmap='gray', vmax=vmax); plt.title('Zero-Filled'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(recon_abs, cmap='gray', vmax=vmax); plt.title('ZS-SSL Recon'); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'recon_visualization.png'), dpi=150); plt.close()


if __name__ == '__main__':
    main()
