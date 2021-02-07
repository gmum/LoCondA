import argparse
import functools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tqdm
import trimesh
import trimesh.repair
from datasets.pointflow import cate_to_synsetid
from watertightness import calc_watertightness_trimesh

FORMAT = ".obj"
logging.disable()


def as_mesh(
    scene_or_mesh: Union[trimesh.Trimesh, trimesh.Scene]
) -> trimesh.Trimesh:
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            data = []
            for g in scene_or_mesh.geometry.values():
                part = trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                trimesh.repair.fix_inversion(part, multibody=True)
                data.append(part)
            mesh = trimesh.util.concatenate(tuple(data))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def get_watertightness(
    reconstructed_shapes_folder: Path, class_name: str, object_name: str
) -> float:
    pred_shape = (
        reconstructed_shapes_folder / class_name / "val" / object_name
    ).with_suffix(FORMAT)
    watertightness = calc_watertightness_trimesh(
        trimesh.load(pred_shape.as_posix()), eps=1e-3
    )
    return watertightness.item()


def validate_paths(
    raw_shapenet_data_folder: Path,
    reconstructed_shapes_folder: Path,
    class_name: str,
):
    shapenet_files = set(
        [
            file.parent.parent.name
            for file in (raw_shapenet_data_folder / class_name).rglob(
                "model_normalized.obj"
            )
        ]
    )
    reconstructed_shapes = set(
        [
            file.with_suffix("").name
            for file in (reconstructed_shapes_folder / class_name).rglob(
                "*" + FORMAT
            )
        ]
    )

    assert (
        len(reconstructed_shapes - shapenet_files) == 0
    ), "Shapenet shapes and reconstructed shape do not match"


def get_objects(reconstructions_path: Path, class_name: str) -> List[str]:
    return [
        file.with_suffix("").name
        for file in (reconstructions_path / class_name).rglob("*.obj")
    ]


def measure_watertigthness(
    shapenet_folder: str, to_compare_folders: List[str], classes: List[str]
):
    shapenet_folder_path = Path(shapenet_folder)
    to_compare_folders_paths = [Path(path) for path in to_compare_folders]
    for kls in classes:
        for folder in to_compare_folders_paths:
            validate_paths(shapenet_folder_path, folder, kls)

    metrics_watertightness: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for to_compare in to_compare_folders_paths:
        print("Comparing: {}".format(to_compare.as_posix()))
        with tqdm.tqdm(classes) as pbar:
            for kls in pbar:
                pbar.set_description("Class: {}".format(kls))

                kls_id = cate_to_synsetid[kls]
                objects = get_objects(to_compare, kls_id)

                partial_get_watertigthness = functools.partial(
                    get_watertightness, to_compare, kls_id
                )
                with tqdm.tqdm(total=len(objects)) as pbar_obj:
                    for watertightness in map(
                        partial_get_watertigthness, objects
                    ):
                        metrics_watertightness[to_compare.name][kls].append(
                            watertightness
                        )
                        pbar_obj.update(1)
                        pbar.set_description_str(
                            "Last watertightness: {:.4f}".format(
                                watertightness
                            )
                        )
    final_metrics_watertightness: Dict[str, Dict[str, float]] = {}
    for folder_name in metrics_watertightness.keys():
        final_metrics_watertightness[folder_name] = {}
        for kls, values in metrics_watertightness[folder_name].items():
            final_metrics_watertightness[folder_name][kls] = np.mean(values)

        final_metrics_watertightness[folder_name]["average"] = np.mean(
            [
                final_metrics_watertightness[folder_name][kls]
                for kls in metrics_watertightness[folder_name].keys()
            ]
        )

    print(final_metrics_watertightness)

    with open("watertightness-results.json", "w") as f:
        json.dump(final_metrics_watertightness, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapenet",
        required=True,
        help="Path to the directory with ShapeNetCore.v2 meshes",
        type=str,
    )

    parser.add_argument(
        "--to_compare",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Multiple paths to different folders where reconstructed meshes "
            "are located"
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        type=str,
        help="Classes to be used to compare",
    )

    args = parser.parse_args()
    measure_watertigthness(args.shapenet, args.to_compare, args.classes)


if __name__ == "__main__":
    main()
