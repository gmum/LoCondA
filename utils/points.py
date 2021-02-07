import torch
import numpy as np


def generate_points_from_normal(shape, mean=0, std=1):
    return torch.empty(shape).normal_(mean, std)


def generate_points_from_uniform_distribution(size, low=-1, high=1, norm=True):
    if norm:
        while True:
            points = torch.zeros([size[0] * 3, *size[1:]]).uniform_(low, high)
            points = points[torch.norm(points, dim=1) < 1]
            if points.shape[0] >= size[0]:
                return points[:size[0]]
    else:
        return torch.zeros([size[0], *size[1:]]).uniform_(low, high)


def generate_points(config, epoch, size, normalize_points=None):
    if config['target_network_input']['normalization']['sample_from_normal']:
        return generate_points_from_normal(size)

    if normalize_points is None:
        normalize_points = config['target_network_input']['normalization']['enable']

    if normalize_points and config['target_network_input']['normalization']['type'] == 'progressive':
        normalization_max_epoch = config['target_network_input']['normalization']['epoch']

        normalization_coef = np.linspace(0, 1, normalization_max_epoch)[epoch - 1] \
            if epoch <= normalization_max_epoch else 1
        points = generate_points_from_uniform_distribution(size=size)
        points[np.linalg.norm(points, axis=1) < normalization_coef] = \
            normalization_coef * (
                    points[
                        np.linalg.norm(points, axis=1) < normalization_coef].T /
                    torch.from_numpy(
                        np.linalg.norm(points[np.linalg.norm(points, axis=1) < normalization_coef], axis=1)).float()
            ).T
    else:
        points = generate_points_from_uniform_distribution(size=size)

    return points


# MIT License
# https://github.com/ThibaultGROUEIX/AtlasNet/blob/2baafa5607d6ee3acb5fffaaae100e5efc754f99/model/template.py#L91
def generate_square(grain):
    """
    Generate a square mesh from a regular grid.
    :param grain:
    Author : Thibault Groueix 01.11.2019
    """
    grain = int(grain)
    grain = grain - 1  # to return grain*grain points
    # generate regular grid
    faces = []
    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])

    for i in range(1, int(grain + 1)):
        for j in range(0, (int(grain + 1) - 1)):
            faces.append([j + (grain + 1) * i,
                          j + (grain + 1) * i + 1,
                          j + (grain + 1) * (i - 1)])
    for i in range(0, (int((grain + 1)) - 1)):
        for j in range(1, int((grain + 1))):
            faces.append([j + (grain + 1) * i,
                          j + (grain + 1) * i - 1,
                          j + (grain + 1) * (i + 1)])

    return torch.tensor(vertices), torch.tensor(faces)
