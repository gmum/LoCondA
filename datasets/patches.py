import math
from typing import Tuple
from utils.points import generate_square

import numpy as np
import torch


class PatchesProcessor:
    def __init__(self, points_per_side: int, total_patches: int, seed: int = 0xCAFFE) -> None:
        self.total_patches = total_patches
        self.rng = np.random.RandomState(seed)

        patch_vertices = []
        patch_edges = []

        unit = 0.5 / points_per_side
        base_vertices, base_edges = generate_square(points_per_side)

        for _ in range(total_patches):
            offsets = self.rng.rand(*base_vertices.shape)
            offsets_abs = np.abs(offsets)
            a_coeff = offsets_abs[:, 1] / offsets_abs[:, 0]
            new_x = unit / (a_coeff + 1)
            new_y = new_x * a_coeff

            lengths = np.linalg.norm(np.stack((new_x, new_y), axis=-1), axis=-1, keepdims=True)
            new_lengths = np.linalg.norm(offsets, axis=-1, keepdims=True) * lengths
            direction = self.rng.randn(*base_vertices.shape)

            new_offsets = (direction / np.linalg.norm(offsets, axis=-1, keepdims=True) * new_lengths)
            new_vertices = base_vertices + new_offsets

            patch_vertices.append(new_vertices)
            patch_edges.append(base_edges)

        self.patch_vertices = torch.stack(patch_vertices, dim=0).float()
        self.patch_edges = torch.stack(patch_edges, dim=0).long()

    def sample_patches_vertices(self, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.rng.randint(0, self.total_patches, size=num)
        return self.patch_vertices[indices], torch.from_numpy(indices).to(self.patch_vertices).long()

    def calculate_total_cosine_distance_of_normals(self, points: torch.Tensor, indices: torch.Tensor,
                                                   device: torch.device) -> torch.Tensor:
        patch_edges = self.patch_edges[indices]
        triangle_points = (
            points.unsqueeze(dim=1)
            .repeat((1, patch_edges.shape[1], 1, 1))
            .gather(
                index=patch_edges.unsqueeze(dim=-1).repeat((1, 1, 1, 3)).to(device),
                dim=-2,
            )
        )

        normals = torch.cross(
            triangle_points[:, :, 1] - triangle_points[:, :, 0],
            triangle_points[:, :, 2] - triangle_points[:, :, 0],
        )

        normalized_normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-5)
        distances = (1 - torch.bmm(normalized_normals, normalized_normals.permute((0, 2, 1))).abs()).tril(-1)
        total_cosine_distance = distances.pow(2).sum(dim=(1, 2))

        return total_cosine_distance
