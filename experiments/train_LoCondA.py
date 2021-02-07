import argparse
import json
import logging
from itertools import chain
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader

from datasets.patches import PatchesProcessor
from models import aae
import torch.optim as optim
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import shutil
import os
from scipy.spatial import Delaunay

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed, get_weights_dir
from utils.points import generate_points, generate_points_from_uniform_distribution, generate_square

cudnn.benchmark = True


def main(config):
    set_seed(config['seed'])

    regularization_experiment = '_edge_length_regularization' if config['LoCondA']['edge_length_regularization']['use']\
        else ('_regularize_normal_deviations' if config['LoCondA']['regularize_normal_deviations']['use'] else '')
    results_dir = prepare_results_dir(config, config['arch'],
                                      'atlas_training' + ('_atlas_net_tn' if config['LoCondA']['use_AtlasNet_TN']
                                                          else '') + regularization_experiment)
    starting_epoch = find_latest_epoch(results_dir) + 1

    hc_weights_path = get_weights_dir(config)
    hc_epoch = find_latest_epoch(hc_weights_path)

    if not hc_epoch:
        print("Invalid 'weights_path' in configuration")
        exit(1)

    if config['LoCondA']['edge_length_regularization']['use'] and \
            config['LoCondA']['regularize_normal_deviations']['use']:
        print("Cannot be used both regularization")
        exit(1)

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger('vae')

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == 'pointflow':
        from datasets.pointflow import ShapeNet15kPointClouds
        dataset = ShapeNet15kPointClouds(root_dir=config['data_dir'],
                                         categories=config['classes'],
                                         tr_sample_size=config['n_points'],
                                         te_sample_size=config['n_points'],
                                         disable_normalization=config['disable_normalization'],
                                         recenter_per_shape=config['recenter_per_shape'],
                                         normalize_per_shape=config['normalize_per_shape'],
                                         normalize_std_per_axis=config['normalize_std_per_axis'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'], drop_last=True,
                                   pin_memory=True)

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)
    atlas_target_network_class = aae.AtlasNetTargetNetwork if config['LoCondA']['use_AtlasNet_TN'] else aae.TargetNetwork

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    log.info(f"Loading E, HN weights for epoch {hc_epoch}...")
    hyper_network.load_state_dict(torch.load(join(hc_weights_path, f'{hc_epoch:05}_G.pth')))
    encoder.load_state_dict(torch.load(join(hc_weights_path, f'{hc_epoch:05}_E.pth')))

    # freeze models learning
    for p in chain(encoder.parameters(), hyper_network.parameters()):
        p.requires_grad = False
    hyper_network.eval()
    encoder.eval()

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading Atlas TN weights...")
        atlas_hyper_network = torch.load(join(weights_path, f'{starting_epoch - 1:05}_A.pth'))
        if not isinstance(atlas_hyper_network, torch.nn.Module):
            model = aae.HyperNetwork(config, device, 5, config['LoCondA']['use_AtlasNet_TN']).to(device)
            model.load_state_dict(atlas_hyper_network)
            atlas_hyper_network = model

        log.info("Loading losses...")
        losses_r = np.load(join(metrics_path, f'{starting_epoch - 1:05}_R.npy')).tolist()
    else:
        atlas_hyper_network = aae.HyperNetwork(config, device, 5, config['LoCondA']['use_AtlasNet_TN']).to(device)
        log.info("First epoch")
        losses_r = []

    #
    # Optimizers
    #
    atlas_hn_optimizer = getattr(optim, config['optimizer']['Atlas_HN']['type'])
    atlas_hn_optimizer = atlas_hn_optimizer(atlas_hyper_network.parameters(),
                                            **config['optimizer']['Atlas_HN']['hyperparams'])
    if starting_epoch > 1:
        log.info("Loading Atlas Optimizer weights...")
        atlas_hn_optimizer.load_state_dict(torch.load(join(weights_path, f'{starting_epoch - 1:05}_Ao.pth')))

    patch_points = config['LoCondA']['grain'] ** 2
    if config['LoCondA']['edge_length_regularization']['use'] and \
            not config['LoCondA']['edge_length_regularization']['random_grid']:
        patch_points = config['LoCondA']['edge_length_regularization']['grain'] ** 2
    elif config['LoCondA']['regularize_normal_deviations']['use']:
        patch_points = config['LoCondA']['regularize_normal_deviations']['grain'] ** 2

    patch_dataset = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)
        atlas_hyper_network.train()

        total_loss_rec = 0.0
        total_loss_reg = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            if dataset_name == 'pointflow':
                X = point_data['tr_points']
            else:
                X, _ = point_data

            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes, _, _ = encoder(X)
            target_networks_weights = hyper_network(codes)
            atlas_target_networks_weights = atlas_hyper_network(codes)
            reconstruction_points = config['LoCondA'].get('reconstruction_points', X.shape[2])

            atlas_nearest_points = torch.zeros(X.shape[0], reconstruction_points * patch_points, X.shape[1]).to(device)
            atlas_rec = torch.zeros(X.shape[0], reconstruction_points * patch_points, X.shape[1]).to(device)
            regularization_loss = torch.zeros(X.shape[0], reconstruction_points).to(device)
            for j, (target_network_weights, atlas_target_network_weights) in enumerate(
                    zip(target_networks_weights, atlas_target_networks_weights)):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)
                target_network_input = generate_points(config=config, epoch=hc_epoch, size=(reconstruction_points,
                                                                                            X.shape[1]))
                X_rec = target_network(target_network_input.to(device))
                x_rec_kneighbors = X_rec

                clf = KNeighborsClassifier(patch_points + 1)
                clf.fit(X_rec.cpu().numpy(), np.ones(len(X_rec)))

                atlas_target_network = atlas_target_network_class(config, atlas_target_network_weights, 5).to(device)
                atlas_input = generate_points_from_uniform_distribution(size=(X_rec.shape[0] * patch_points, 2),
                                                                        low=0, high=1, norm=False)
                if config['LoCondA']['edge_length_regularization']['use']:
                    if config['LoCondA']['edge_length_regularization']['random_grid']:
                        faces = torch.zeros(X_rec.shape[0], 2 * patch_points - 2, X_rec.shape[1])
                        for z, points in enumerate(atlas_input.reshape(X_rec.shape[0], patch_points, 2)):
                            simplices = torch.from_numpy(Delaunay(points).simplices)
                            faces[z, :simplices.shape[0]] = simplices
                    else:
                        grain = config['LoCondA']['edge_length_regularization']['grain']
                        vertices, faces = generate_square(grain)
                        atlas_input = vertices.repeat(X_rec.shape[0], 1)
                        faces = faces.repeat(X_rec.shape[0], 1).reshape(X_rec.shape[0], faces.shape[0], 3)
                elif config['LoCondA']['regularize_normal_deviations']['use']:
                    if patch_dataset is None:
                        log.info('Generating patch_dataset')
                        patch_dataset = PatchesProcessor(
                            config['LoCondA']['regularize_normal_deviations']['grain'],
                            config['LoCondA']['regularize_normal_deviations']['num_patches']
                        )
                    atlas_input, faces = patch_dataset.sample_patches_vertices(X_rec.shape[0])
                    atlas_input = atlas_input.reshape(X_rec.shape[0] * patch_points, 2)

                atlas_target_network_input = torch.cat([
                    target_network_input[:, None, :].expand(X_rec.shape[0], patch_points, X_rec.shape[1]).reshape(
                        X_rec.shape[0] * patch_points, X_rec.shape[1]),
                    atlas_input
                ], 1).to(device)

                nearest_points = clf.kneighbors(x_rec_kneighbors.cpu().numpy(), return_distance=False)
                x_rec_nearest_points = X_rec[nearest_points[:, 1:].reshape(-1)]

                atlas_rec[j] = atlas_target_network(atlas_target_network_input.to(device))
                atlas_nearest_points[j] = x_rec_nearest_points

                if config['LoCondA']['edge_length_regularization']['use']:
                    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, (0, 2)]], 1).long()
                    vertices_rec = atlas_rec[j].reshape(X_rec.shape[0], patch_points, X_rec.shape[1])
                    regularization_loss[j] = torch.sum(
                        torch.norm(vertices_rec[-1, edges[:, :, 0], :] - vertices_rec[-1, edges[:, :, 1], :], dim=2),
                        dim=1)
                elif config['LoCondA']['regularize_normal_deviations']['use']:
                    vertices_rec = atlas_rec[j].clone().reshape((X_rec.shape[0], patch_points, X_rec.shape[1]))
                    regularization_loss[j] = patch_dataset.calculate_total_cosine_distance_of_normals(vertices_rec,
                                                                                                      faces,
                                                                                                      device)

            losses = torch.zeros(patch_points).to(device)
            for bs in range(patch_points):
                start, end = bs * reconstruction_points, (bs + 1) * reconstruction_points
                losses[bs] = reconstruction_loss(
                    atlas_nearest_points[:, start:end, :],
                    atlas_rec[:, start:end, :]
                )

            loss_rec = config['LoCondA']['reconstruction_coef'] * losses.mean()
            loss_reg = config['LoCondA']['regularization_coef'] * regularization_loss.mean()

            atlas_hn_optimizer.zero_grad()
            encoder.zero_grad()
            hyper_network.zero_grad()
            atlas_hyper_network.zero_grad()

            total_loss = loss_rec + loss_reg
            total_loss.backward()
            atlas_hn_optimizer.step()

            total_loss_rec += loss_rec.item()
            total_loss_reg += loss_reg.item()

        log.info(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_All: {(total_loss_rec + total_loss_reg) / i:.4f} '
            f'Loss_Reconstruction: {total_loss_rec / i:.4f} '
            f'Loss_Regularization: {total_loss_reg / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )

        losses_r.append(total_loss_rec)

        #
        # Save intermediate results
        #
        if epoch % config['save_samples_frequency'] == 0:
            log.debug('Saving samples...')

            X = X[-1].cpu().numpy()
            X_rec = X_rec.detach().cpu().numpy()
            atlas_rec = atlas_rec[-1].detach().cpu().numpy()

            for m in range(3):
                fig = plot_3d_point_cloud(X_rec[:, 0], X_rec[:, 1], X_rec[:, 2], in_u_sphere=True, show=False,
                                          x1=[X_rec[:, 0][m]], y1=[X_rec[:, 1][m]], z1=[X_rec[:, 2][m]])
                fig.savefig(join(results_dir, 'samples', f'{epoch}_{m}_reconstructed.png'))
                plt.close(fig)

                fig = plot_3d_point_cloud(X_rec[:, 0], X_rec[:, 1], X_rec[:, 2], in_u_sphere=True, show=False,
                                          x1=atlas_rec[m*patch_points:(m+1)*patch_points, 0],
                                          y1=atlas_rec[m*patch_points:(m+1)*patch_points, 1],
                                          z1=atlas_rec[m*patch_points:(m+1)*patch_points, 2])
                fig.savefig(join(results_dir, 'samples', f'{epoch}_{m}_atlas.png'))
                plt.close(fig)

            fig = plot_3d_point_cloud(X[0], X[1], X[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'samples', f'{epoch}_real.png'))
            plt.close(fig)

        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['save_weights_frequency'] == 0:
            log.debug('Saving weights...')

            torch.save(atlas_hyper_network, join(weights_path, f'{epoch:05}_A.pth'))
            torch.save(atlas_hn_optimizer.state_dict(), join(weights_path, f'{epoch:05}_Ao.pth'))

            np.save(join(metrics_path, f'{epoch:05}_R'), np.array(losses_r))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
