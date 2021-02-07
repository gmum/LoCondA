import argparse
import json
import logging
import pickle
from os.path import join, exists
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

from datasets.patches import PatchesProcessor
from models import aae
from utils.sphere_triangles import generate
import math

from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed, get_weights_dir
from utils.points import generate_points, generate_points_from_uniform_distribution, generate_square

cudnn.benchmark = True


def main(config):
    set_seed(config['seed'])

    regularization_experiment = '_edge_length_regularization' if config['edge_length_regularization'] \
        else ('_regularize_normal_deviations' if config['regularize_normal_deviations'] else '')
    results_dir = prepare_results_dir(config, config['arch'],
                                      'atlas_experiments' + ('_atlas_net_tn' if config['use_AtlasNet_TN']
                                                             else '') + regularization_experiment,
                                      dirs_to_create=['points_interpolation', 'reconstruction', 'square_mesh', 'fixed',
                                                      'interpolation'])
    hc_weights_path = get_weights_dir(config)
    hc_epoch = find_latest_epoch(hc_weights_path)

    weights_path = get_weights_dir(config, experiment='atlas_training' + ('_atlas_net_tn' if config['use_AtlasNet_TN']
                                                                          else '') + regularization_experiment)
    epoch = find_latest_epoch(weights_path)

    if not hc_epoch:
        print("Invalid 'weights_path' in configuration")
        exit(1)

    if config['edge_length_regularization'] and config['regularize_normal_deviations']:
        print("Cannot be used both regularization")
        exit(1)

    setup_logging(results_dir)
    global log
    log = logging.getLogger()

    if not exists(join(results_dir, 'experiment_config.json')):
        with open(join(results_dir, 'experiment_config.json'), mode='w') as f:
            json.dump(config, f)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'],
                                  split='valid')
    elif dataset_name == 'pointflow':
        from datasets.pointflow import ShapeNet15kPointClouds
        dataset = ShapeNet15kPointClouds(root_dir=config['data_dir'],
                                         categories=config['classes'],
                                         tr_sample_size=config['n_points'],
                                         te_sample_size=config['n_points'],
                                         split='val',
                                         disable_normalization=config['disable_normalization'],
                                         recenter_per_shape=config['recenter_per_shape'],
                                         normalize_per_shape=config['normalize_per_shape'],
                                         normalize_std_per_axis=config['normalize_std_per_axis'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']), len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)
    atlas_target_network_class = aae.AtlasNetTargetNetwork if config['use_AtlasNet_TN'] else aae.TargetNetwork

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    log.info(f"Loading weights for epoch {epoch}...")
    atlas_hyper_network = torch.load(join(weights_path, f'{epoch:05}_A.pth'))
    if not isinstance(atlas_hyper_network, torch.nn.Module):
        model = aae.HyperNetwork(config, device, 5, config['use_AtlasNet_TN']).to(device)
        model.load_state_dict(atlas_hyper_network)
        atlas_hyper_network = model

    log.info(f"Loading E, HN weights for epoch {hc_epoch}...")
    hyper_network.load_state_dict(torch.load(join(hc_weights_path, f'{hc_epoch:05}_G.pth')))
    encoder.load_state_dict(torch.load(join(hc_weights_path, f'{hc_epoch:05}_E.pth')))

    hyper_network.eval()
    encoder.eval()
    atlas_hyper_network.eval()

    x = []
    idx = []
    total_loss = 0.0
    patch_points = 10
    with torch.no_grad():
        for i, point_data in enumerate(points_dataloader, 1):
            if dataset_name == 'pointflow':
                X = point_data['tr_points']
                cat_id, file_id = point_data['sid'], point_data['mid']
            else:
                X, (cat_id, file_id) = point_data

            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            x.append(X)
            idx += list(zip(cat_id, file_id))

            codes, mu, logvar = encoder(X)
            target_networks_weights = hyper_network(codes)
            atlas_target_networks_weights = atlas_hyper_network(codes)

            atlas_nearest_points = torch.zeros(X.shape[0], X.shape[2] * patch_points, X.shape[1]).to(device)
            atlas_rec = torch.zeros(X.shape[0], X.shape[2] * patch_points, X.shape[1]).to(device)
            for j, (target_network_weights, atlas_target_network_weights) in enumerate(
                    zip(target_networks_weights, atlas_target_networks_weights)):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)
                target_network_input = generate_points(config=config, epoch=hc_epoch, size=(X.shape[2], X.shape[1]))
                X_rec = target_network(target_network_input.to(device))

                clf = KNeighborsClassifier(patch_points + 1)
                clf.fit(X_rec.cpu().numpy(), np.ones(len(X_rec)))

                atlas_target_network = atlas_target_network_class(config, atlas_target_network_weights, 5).eval().to(device)

                atlas_target_network_input = torch.cat([
                    target_network_input[:, None, :].expand(X_rec.shape[0], patch_points, X_rec.shape[1]).reshape(
                        X_rec.shape[0] * patch_points, X_rec.shape[1]),
                    generate_points_from_uniform_distribution(size=(X_rec.shape[0] * patch_points, 2),
                                                              low=0, high=1, norm=False)
                ], 1).to(device)

                nearest_points = clf.kneighbors(X_rec.cpu().numpy(), return_distance=False)
                x_rec_nearest_points = X_rec[nearest_points[:, 1:].reshape(-1)]

                atlas_rec[j] = atlas_target_network(atlas_target_network_input.to(device))
                atlas_nearest_points[j] = x_rec_nearest_points

            losses = torch.zeros(patch_points).to(device)
            for bs in range(patch_points):
                start, end = bs * X.shape[2], (bs + 1) * X.shape[2]
                losses[bs] = reconstruction_loss(
                    atlas_nearest_points[:, start:end, :],
                    atlas_rec[:, start:end, :]
                )

            loss_r = config['reconstruction_coef'] * losses.mean()
            total_loss += loss_r.item()

        log.info(
            f'Loss_ALL: {total_loss / i:.4f} '
        )

        x = torch.cat(x)

        if config['experiments']['interpolation_between_two_points']['execute']:
            interpolation_between_two_points(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class,
                                             device, x, idx, results_dir, hc_epoch,
                                             config['experiments']['interpolation_between_two_points']['amount'],
                                             config['experiments']['interpolation_between_two_points']['sphere_triangles_points'],
                                             config['experiments']['interpolation_between_two_points']['square_mesh'],
                                             config['experiments']['interpolation_between_two_points']['image_points'],
                                             config['experiments']['interpolation_between_two_points']['patch_points'],
                                             config['experiments']['interpolation_between_two_points']['transitions'])

        if config['experiments']['reconstruction']['execute']:
            reconstruction(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, idx,
                           results_dir, hc_epoch,
                           config['experiments']['reconstruction']['amount'],
                           config['experiments']['reconstruction']['sphere_triangles_points'],
                           config['experiments']['reconstruction']['image_points'],
                           config['experiments']['reconstruction']['patch_points'])

        if config['experiments']['square_mesh']['execute']:
            square_mesh(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, idx,
                        results_dir,
                        hc_epoch,
                        config['experiments']['square_mesh']['amount'],
                        config['experiments']['square_mesh']['sphere_triangles_points'],
                        config['experiments']['square_mesh']['grain'],
                        config['experiments']['square_mesh']['image_points'])

        if config['experiments']['fixed']['execute']:
            fixed(hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, results_dir, hc_epoch,
                  config['experiments']['fixed']['amount'],
                  config['z_size'],
                  config['experiments']['fixed']['mean'],
                  config['experiments']['fixed']['std'],
                  config['experiments']['fixed']['sphere_triangles_points'],
                  config['experiments']['fixed']['square_mesh'],
                  config['experiments']['fixed']['image_points'],
                  config['experiments']['fixed']['patch_points'])

        if config['experiments']['interpolation']['execute']:
            interpolation(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x,
                          results_dir, hc_epoch,
                          config['experiments']['interpolation']['amount'],
                          config['experiments']['interpolation']['sphere_triangles_points'],
                          config['experiments']['interpolation']['square_mesh'],
                          config['experiments']['interpolation']['image_points'],
                          config['experiments']['interpolation']['patch_points'],
                          config['experiments']['interpolation']['transitions'])


def interpolation_between_two_points(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x,
                                     idx, results_dir, epoch, amount, sphere_triangles_points, _square_mesh, image_points,
                                     patch_points, transitions):
    log.info("Interpolations between two points")
    amount = len(x) if amount == 'all' else amount
    X = x[:amount]
    log.info(f"{len(X)} samples")
    bs = 16

    for i, x in enumerate(DataLoader(X, batch_size=bs, drop_last=False, shuffle=False)):
        codes, _, _ = encoder(x)
        target_networks_weights = hyper_network(codes)
        atlas_target_networks_weights = atlas_hyper_network(codes)
        x = x.cpu().numpy()
        for k in range(len(x)):
            y = i * bs + k
            target_network = aae.TargetNetwork(config, target_networks_weights[k])
            if sphere_triangles_points['use']:
                method = sphere_triangles_points['method']
                depth = sphere_triangles_points['depth']
                target_network_input, triangulation = generate(method, depth)
                with open(join(results_dir, 'points_interpolation', f'{y}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)
            else:
                target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
            x_rec = target_network(target_network_input.to(device)).cpu().numpy()

            x_a = target_network_input[torch.argmin(target_network_input, dim=0)[2]][None, :]
            x_b = target_network_input[torch.argmax(target_network_input, dim=0)[2]][None, :]
            interpolation_input = torch.zeros(transitions, x.shape[1])
            for j, alpha in enumerate(np.linspace(0, 1, transitions)):
                interpolation_input[j] = (1 - alpha) * x_a + alpha * x_b  # interpolate point
            interpolation_rec = target_network(interpolation_input.to(device))

            if _square_mesh['use']:
                grain = _square_mesh['grain']
                patch_points = grain ** 2
                vertices, faces = generate_square(grain)
                np.save(join(results_dir, 'points_interpolation', f'{y}_atlas_triangulation'), np.array(faces))
                atlas_input = vertices.repeat(transitions, 1)
            elif config['regularize_normal_deviations']:
                grain = int(math.sqrt(patch_points))
                patch_points = grain ** 2
                patch_dataset = PatchesProcessor(grain, config['num_patches'])
                atlas_input, faces = patch_dataset.sample_patches_vertices(transitions)
                np.save(join(results_dir, 'points_interpolation', f'{y}_atlas_triangulation'), np.array(faces))
                atlas_input = atlas_input.reshape(transitions * patch_points, 2)
            else:
                atlas_input = generate_points_from_uniform_distribution(size=(transitions * patch_points, 2), low=0,
                                                                        high=1, norm=False)

            atlas_target_network = atlas_target_network_class(config, atlas_target_networks_weights[k], 5).eval()
            atlas_target_network_input = torch.cat([
                interpolation_input[:, None, :].expand(transitions, patch_points, x.shape[1]).reshape(
                    transitions * patch_points, x.shape[1]),
                atlas_input
            ], 1)
            atlas_rec = atlas_target_network(atlas_target_network_input.to(device)).reshape(transitions, patch_points,
                                                                                            x.shape[1])

            np.save(join(results_dir, 'points_interpolation', f'{y}_target_network_input'), target_network_input.cpu().numpy())
            np.save(join(results_dir, 'points_interpolation', f'{y}_reconstruction'), x_rec)
            np.save(join(results_dir, 'points_interpolation', f'{y}_points_interpolation'), interpolation_rec.cpu().numpy())
            np.save(join(results_dir, 'points_interpolation', f'{y}_atlas_interpolation'), atlas_rec.cpu().numpy())
            np.save(join(results_dir, 'points_interpolation', f'{y}_real'), x[k])
            with open(join(results_dir, 'points_interpolation', f'{y}_dataset_info.json'), 'w') as info_file:
                json.dump({'category_id': idx[y][0], 'filename': idx[y][1]}, info_file)


def reconstruction(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, idx, results_dir,
                   epoch, amount, sphere_triangles_points, image_points, patch_points):
    log.info("Reconstruction")
    amount = len(x) if amount == 'all' else amount
    X = x[:amount]
    log.info(f"{len(X)} samples")
    bs = 16

    for i, x in enumerate(DataLoader(X, batch_size=bs, drop_last=False, shuffle=False)):
        codes, _, _ = encoder(x)
        target_networks_weights = hyper_network(codes)
        atlas_target_networks_weights = atlas_hyper_network(codes)
        x = x.cpu().numpy()

        for k in range(len(x)):
            y = i * bs + k
            target_network = aae.TargetNetwork(config, target_networks_weights[k])
            if sphere_triangles_points['use']:
                method = sphere_triangles_points['method']
                depth = sphere_triangles_points['depth']
                target_network_input, triangulation = generate(method, depth)
                image_points = target_network_input.shape[0]
                with open(join(results_dir, 'reconstruction', f'{y}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)
            else:
                target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
            x_rec = target_network(target_network_input.to(device)).cpu().numpy()

            atlas_target_network = atlas_target_network_class(config, atlas_target_networks_weights[k], 5).eval()
            atlas_target_network_input = torch.cat([
                target_network_input[:, None, :].expand(image_points, patch_points, x.shape[1]).reshape(
                    image_points * patch_points, x.shape[1]),
                generate_points_from_uniform_distribution(size=(image_points * patch_points, 2), low=0, high=1, norm=False)
            ], 1)
            atlas_rec = atlas_target_network(atlas_target_network_input.to(device)).reshape(image_points, patch_points,
                                                                                            x.shape[1])

            np.save(join(results_dir, 'reconstruction', f'{y}_target_network_input'), target_network_input.cpu().numpy())
            np.save(join(results_dir, 'reconstruction', f'{y}_real'), x[k])
            np.save(join(results_dir, 'reconstruction', f'{y}_reconstructed'), x_rec)
            np.save(join(results_dir, 'reconstruction', f'{y}_atlas'), atlas_rec.cpu().numpy())
            np.save(join(results_dir, 'reconstruction', f'{y}_atlas_target_network_input'),
                    atlas_target_network_input.cpu().numpy())
            with open(join(results_dir, 'reconstruction', f'{y}_dataset_info.json'), 'w') as info_file:
                json.dump({'category_id': idx[y][0], 'filename': idx[y][1]}, info_file)


def square_mesh(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, idx, results_dir, epoch,
                amount, sphere_triangles_points, grain, image_points):
    log.info("Square mesh")
    amount = len(x) if amount == 'all' else amount
    X = x[:amount]
    log.info(f"{len(X)} samples")
    bs = 16

    for i, x in enumerate(DataLoader(X, batch_size=bs, drop_last=False, shuffle=False)):
        codes, _, _ = encoder(x)
        target_networks_weights = hyper_network(codes)
        atlas_target_networks_weights = atlas_hyper_network(codes)
        x = x.cpu().numpy()
        patch_points = grain**2
        for k in range(len(x)):
            y = i * bs + k
            target_network = aae.TargetNetwork(config, target_networks_weights[k])
            if sphere_triangles_points['use']:
                method = sphere_triangles_points['method']
                depth = sphere_triangles_points['depth']
                target_network_input, triangulation = generate(method, depth)
                image_points = target_network_input.shape[0]
                with open(join(results_dir, 'square_mesh', f'{y}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)
            else:
                target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
            x_rec = target_network(target_network_input.to(device)).cpu().numpy()

            atlas_target_network = atlas_target_network_class(config, atlas_target_networks_weights[k], 5).eval()
            vertices, faces = generate_square(grain)
            atlas_target_network_input = torch.cat([
                target_network_input[:, None, :].expand(image_points, patch_points, x.shape[1]).reshape(
                    image_points * patch_points, x.shape[1]),
                vertices.repeat(image_points, 1)
            ], 1)
            atlas_rec = atlas_target_network(atlas_target_network_input.to(device)).reshape(image_points, patch_points,
                                                                                            x.shape[1])

            np.save(join(results_dir, 'square_mesh', f'{y}_target_network_input'), target_network_input.cpu().numpy())
            np.save(join(results_dir, 'square_mesh', f'{y}_real'), x[k])
            np.save(join(results_dir, 'square_mesh', f'{y}_reconstructed'), x_rec)
            np.save(join(results_dir, 'square_mesh', f'{y}_atlas'), atlas_rec.cpu().numpy())
            np.save(join(results_dir, 'square_mesh', f'{y}_atlas_target_network_input'),
                    atlas_target_network_input.cpu().numpy())
            np.save(join(results_dir, 'square_mesh', f'{y}_atlas_triangulation'), faces.numpy())
            with open(join(results_dir, 'square_mesh', f'{y}_dataset_info.json'), 'w') as info_file:
                json.dump({'category_id': idx[y][0], 'filename': idx[y][1]}, info_file)


def fixed(hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, results_dir, epoch, amount,
          z_size, fixed_mean, fixed_std, sphere_triangles_points, _square_mesh, image_points, patch_points):
    log.info("Fixed")

    fixed_noise = torch.zeros(amount, z_size).normal_(mean=fixed_mean, std=fixed_std).to(device)
    target_networks_weights = hyper_network(fixed_noise)
    atlas_target_networks_weights = atlas_hyper_network(fixed_noise)

    for k in range(amount):
        target_network = aae.TargetNetwork(config, target_networks_weights[k]).to(device)

        if sphere_triangles_points['use']:
            method = sphere_triangles_points['method']
            depth = sphere_triangles_points['depth']
            target_network_input, triangulation = generate(method, depth)
            image_points = target_network_input.shape[0]
            with open(join(results_dir, 'fixed', f'{k}_triangulation.pickle'), 'wb') as triangulation_file:
                pickle.dump(triangulation, triangulation_file)
        else:
            target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
        x_rec = target_network(target_network_input.to(device)).cpu().numpy()

        if _square_mesh['use']:
            grain = _square_mesh['grain']
            patch_points = grain ** 2
            vertices, faces = generate_square(grain)
            np.save(join(results_dir, 'fixed', f'{k}_atlas_triangulation'), np.array(faces))
            atlas_input = vertices.repeat(image_points, 1)
        elif config['regularize_normal_deviations']:
            grain = int(math.sqrt(patch_points))
            patch_points = grain ** 2
            patch_dataset = PatchesProcessor(grain, config['num_patches'])
            atlas_input, faces = patch_dataset.sample_patches_vertices(image_points)
            np.save(join(results_dir, 'fixed', f'{k}_atlas_triangulation'), np.array(faces))
            atlas_input = atlas_input.reshape(image_points * patch_points, 2)
        else:
            atlas_input = generate_points_from_uniform_distribution(size=(image_points * patch_points, 2), low=0, high=1,
                                                                    norm=False)

        atlas_target_network = atlas_target_network_class(config, atlas_target_networks_weights[k], 5).eval()
        atlas_target_network_input = torch.cat([
            target_network_input[:, None, :].expand(image_points, patch_points, x.shape[1]).reshape(
                image_points * patch_points, x.shape[1]),
            atlas_input
        ], 1)
        atlas_rec = atlas_target_network(atlas_target_network_input.to(device)).reshape(image_points, patch_points,
                                                                                        x.shape[1])

        np.save(join(results_dir, 'fixed', f'{k}_target_network_input'), target_network_input.cpu().numpy())
        np.save(join(results_dir, 'fixed', f'{k}_reconstruction'), x_rec)
        np.save(join(results_dir, 'fixed', f'{k}_atlas'), atlas_rec.cpu().numpy())
        np.save(join(results_dir, 'fixed', f'{k}_atlas_target_network_input'),
                atlas_target_network_input.cpu().numpy())


def interpolation(encoder, hyper_network, atlas_hyper_network, atlas_target_network_class, device, x, results_dir,
                  epoch, amount, sphere_triangles_points, _square_mesh, image_points, patch_points, transitions):
    log.info("Interpolation")

    for k in range(amount):
        x_a = x[None, 2 * k, :, :]
        x_b = x[None, 2 * k + 1, :, :]

        with torch.no_grad():
            z_a, _, _ = encoder(x_a)
            z_b, _, _ = encoder(x_b)

        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * z_a + alpha * z_b  # interpolate in the latent space
            target_networks_weights = hyper_network(z_int)  # decode the interpolated sample
            atlas_target_networks_weights = atlas_hyper_network(z_int)  # decode the interpolated sample

            target_network = aae.TargetNetwork(config, target_networks_weights[0]).to(device)

            if sphere_triangles_points['use']:
                method = sphere_triangles_points['method']
                depth = sphere_triangles_points['depth']
                target_network_input, triangulation = generate(method, depth)
                image_points = target_network_input.shape[0]
                with open(join(results_dir, 'interpolation', f'{k}_{j}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)
            else:
                target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
            x_rec = target_network(target_network_input.to(device)).cpu().numpy()

            if _square_mesh['use']:
                grain = _square_mesh['grain']
                patch_points = grain ** 2
                vertices, faces = generate_square(grain)
                np.save(join(results_dir, 'interpolation', f'{k}_{j}_atlas_triangulation'), np.array(faces))
                atlas_input = vertices.repeat(image_points, 1)
            elif config['regularize_normal_deviations']:
                grain = int(math.sqrt(patch_points))
                patch_points = grain ** 2
                patch_dataset = PatchesProcessor(grain, config['num_patches'])
                atlas_input, faces = patch_dataset.sample_patches_vertices(image_points)
                np.save(join(results_dir, 'interpolation', f'{k}_{j}_atlas_triangulation'), np.array(faces))
                atlas_input = atlas_input.reshape(image_points * patch_points, 2)
            else:
                atlas_input = generate_points_from_uniform_distribution(size=(image_points * patch_points, 2), low=0,
                                                                        high=1,
                                                                        norm=False)

            atlas_target_network = atlas_target_network_class(config, atlas_target_networks_weights[0], 5).eval()
            atlas_target_network_input = torch.cat([
                target_network_input[:, None, :].expand(image_points, patch_points, x.shape[1]).reshape(
                    image_points * patch_points, x.shape[1]),
                atlas_input
            ], 1)
            atlas_rec = atlas_target_network(atlas_target_network_input.to(device)).reshape(image_points, patch_points,
                                                                                            x.shape[1])

            np.save(join(results_dir, 'interpolation', f'{k}_{j}_target_network_input'), target_network_input.cpu().numpy())
            np.save(join(results_dir, 'interpolation', f'{k}_{j}_reconstruction'), x_rec)
            np.save(join(results_dir, 'interpolation', f'{k}_{j}_atlas'), atlas_rec.cpu().numpy())
            np.save(join(results_dir, 'interpolation', f'{k}_{j}_atlas_target_network_input'),
                    atlas_target_network_input.cpu().numpy())

        np.save(join(results_dir, 'interpolation', f'{k}_real_a'), x_a.cpu().numpy())
        np.save(join(results_dir, 'interpolation', f'{k}_real_b'), x_b.cpu().numpy())


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
