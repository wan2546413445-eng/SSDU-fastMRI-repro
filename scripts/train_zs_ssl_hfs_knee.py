import os
import time

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
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--acs_block', type=int, default=24)
    parser.add_argument('--viz_percentile', type=float, default=99.5)
    args = parser.parse_args()
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
    np.int = int
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat(args.data_path)
    kspace_train, sens_maps, original_mask = data['kspace'], data['sens_maps'], data['mask']
    args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB = kspace_train.shape
    kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

    cv_trn_mask, cv_val_mask = utils.uniform_selection(
        kspace_train, original_mask, rho=args.rho_val, small_acs_block=(args.acs_block, args.acs_block)
    )
    remainder_mask, cv_val_mask = np.copy(cv_trn_mask), np.copy(np.complex64(cv_val_mask))

    nw_input_val = utils.sense1(kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)), sens_maps)[np.newaxis]
    ref_kspace_val = (kspace_train * np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)))[np.newaxis]

    nw_input_trn = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    trn_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    for jj in range(args.num_reps):
        trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(
            kspace_train, remainder_mask, rho=args.rho_train, small_acs_block=(args.acs_block, args.acs_block)
        )
        sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        nw_input_trn[jj, ...] = utils.sense1(sub_kspace, sens_maps)

    np.savez(os.path.join(args.output_dir, 'zs_ssl_masks.npz'),
             trn_mask=trn_mask, loss_mask=loss_mask, cv_trn_mask=cv_trn_mask, cv_val_mask=cv_val_mask)

    if args.data_opt == 'Coronal_PD':
        trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))
        trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

    sens_maps = np.tile(sens_maps[np.newaxis], (args.num_reps, 1, 1, 1))
    sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
    ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input_trn = utils.complex2real(nw_input_trn)
    ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))
    nw_input_val = utils.complex2real(nw_input_val)

    train_loader = DataLoader(Dataset(nw_input_trn, trn_mask, loss_mask, sens_maps, ref_kspace), batch_size=args.batchSize, shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(nw_input_val, cv_trn_mask[np.newaxis], cv_val_mask[np.newaxis], sens_maps[0][np.newaxis], ref_kspace_val), batch_size=args.batchSize, shuffle=False, num_workers=0)

    model = UnrollNet.UnrolledNet(args, device=device).to(device)
    loss_fn = MixL1L2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_train_loss, total_val_loss = [], []
    valid_loss_min, ep, val_loss_tracker = np.inf, 0, 0
    best_epoch = -1
    while ep < args.epochs and val_loss_tracker < args.stop_training:
        trn_loss, _ = train(train_loader, model, loss_fn, optimizer, device=device)
        val_loss = validation(val_loader, model, loss_fn, device=device)
        total_train_loss.append(trn_loss)
        total_val_loss.append(val_loss)
        checkpoint = {'epoch': ep, 'valid_loss_min': val_loss, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}
        if val_loss <= valid_loss_min:
            valid_loss_min = val_loss
            best_epoch = ep
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            val_loss_tracker = 0
        else:
            val_loss_tracker += 1
        ep += 1

    test_mask = np.complex64(original_mask)
    nw_input_inference = utils.sense1(kspace_train * np.tile(test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)), np.transpose(sens_maps[0], (1, 2, 0)))
    ref_image = utils.sense1(kspace_train, np.transpose(sens_maps[0], (1, 2, 0)))

    test_loader = DataLoader(Dataset_Inference(utils.complex2real(nw_input_inference[np.newaxis]), test_mask[np.newaxis], test_mask[np.newaxis], sens_maps[0][np.newaxis]),
                             batch_size=args.batchSize, shuffle=False, num_workers=0)

    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state'])
    zs_ssl_recon = utils.real2complex(test(test_loader, model, device).to('cpu').numpy())

    ref_image_abs = np.abs(ref_image)
    zf_image_abs = np.abs(nw_input_inference)
    recon_abs = np.abs(zs_ssl_recon)
    ref_scale = np.max(ref_image_abs) + 1e-12

    sio.savemat(os.path.join(args.output_dir, 'recon_outputs.mat'), {
        'recon': recon_abs,
        'zf': zf_image_abs,
        'ref': ref_image_abs,
        'ref_scale': np.float32(ref_scale),
        'best_epoch': np.int32(best_epoch),
        'seed': np.int32(args.seed),
    })
    sio.savemat(os.path.join(args.output_dir, 'TrainingLog.mat'), {'total_train_loss': np.asarray(total_train_loss), 'total_val_loss': np.asarray(total_val_loss)})

    err_zf = np.abs(zf_image_abs - ref_image_abs)
    err_recon = np.abs(recon_abs - ref_image_abs)
    vmax_main = np.percentile(ref_image_abs, args.viz_percentile)
    vmax_err = np.percentile(err_zf, args.viz_percentile)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    imgs = [ref_image_abs, zf_image_abs, recon_abs, err_zf, err_recon]
    titles = ['Ref', 'ZF', 'Recon', '|ZF-Ref|', '|Recon-Ref|']
    for i, ax in enumerate(axes):
        vmax = vmax_main if i < 3 else vmax_err
        ax.imshow(imgs[i], cmap='gray', vmax=vmax, aspect='equal', interpolation='nearest')
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'recon_visualization.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
