# Modeling 3D Surface Manifolds with a Locally Conditioned Atlas (LoCondA) [[ Paper ]](https://arxiv.org/abs/2102.05984)

## Requirements
- dependencies stored in `requirements.txt`.
- Python 3.6+
- cuda

## PointFlow dataset
PointFlow dataset used in experiments was published [here](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing) by [PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows. Guandao Yang*, Xun Huang*, Zekun Hao, Ming-Yu Liu, Serge Belongie, Bharath Hariharan](https://arxiv.org/abs/1906.12320)

## Installation
If you are using `Conda`:
- run `./install_requirements.sh` 

otherwise:
- install `cudatoolkit` and run `pip install -r requirements.txt`

Then execute:
```
export CUDA_HOME=... # e.g. /var/lib/cuda-10.0/
./build_losses.sh
```

#### Watertightness measure
```
git submodule update --init --recursive
cd pytorch-watertightness
make
```


### Configuration (settings/hyperparams.json, settings/experiments.json):
  - HyperCloud
    - *target_network_input:normalization:type* -> progressive
    - *target_network_input:normalization:epoch* -> epoch for which the progressive normalization, of the points from uniform distribution, ends
  - LoCondA
    - *LoCondA:use_AtlasNet_TN* -> use core AtlasNet ([AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://arxiv.org/abs/1802.05384)) function with input size changed from 2 to 5 as Target Network, otherwise HyperCloud Target Network will be used
    - *LoCondA:reconstruction_points* (optional) -> number of points reconstructed by HyperCloud Target Network
    - regularization of the length of the patch edges (at most one regularization can be enabled):
      - *LoCondA:edge_length_regularization*
      - *LoCondA:regularize_normal_deviations*
    - number of patch points:
      - *LoCondA:grain* ^ 2 -> if regularization is not enabled
      - *LoCondA:edge_length_regularization:grain* ^ 2 -> if edge_length_regularization is enabled
      - *LoCondA:regularize_normal_deviations:grain* ^ 2 -> if regularize_normal_deviations is enabled
  - *reconstruction_loss* -> chamfer | earth_mover
  - *dataset* -> shapenet | pointflow


#### Frequency of saving training data (settings/hyperparams.json)
```
"save_weights_frequency": int (> 0) -> save model's weights every x epochs
"save_samples_frequency": int (> 0) -> save intermediate reconstructions every x epochs
```


### HyperCloud Target Network input
#### Uniform distribution:
3D points are sampled from uniform distribution. 

##### Normalization
When normalization is enabled, points are normalized progressively 
from first epoch to `target_network_input:normalization:epoch` epoch specified in the configuration. 

As a result, for epochs >= `target_network_input:normalization:epoch`, target network input is sampled from a uniform unit 3D ball 

Exemplary config:
```
"target_network_input": {
    "constant": false,
    "normalization": {
        "enable": true,
        "type": "progressive",
        "epoch": 100
    }
}
For epochs: [1, 100] target network input is normalized progressively
For epochs: [100, inf] target network input is sampled from a uniform unit 3D ball
``` 


## Usage
**Add project root directory to PYTHONPATH**

```export PYTHONPATH=project_path:$PYTHONPATH```

### Training

#### HyperCloud
- `python experiments/train_HyperCloud.py --config settings/hyperparams.json`

Results will be saved in the directory: `${results_root}/vae/training/uniform*/${dataset}/${classes}`

#### LoCondA
- `python experiments/train_LoCondA.py --config settings/hyperparams.json`

Results will be saved in the directory: 

`${results_root}/vae/atlas_training[_atlas_net_tn][_edge_length_regularization|_regularize_normal_deviations]/uniform*/${dataset}/${classes}`

- `atlas_net_tn` will be added to results path if `use_AtlasNet_TN == True`
- `_edge_length_regularization` will be added to results path if `edge_length_regularization` is enabled
- `_regularize_normal_deviations` will be added to results path if `regularize_normal_deviations` is enabled

HyperCloud model weights are loaded from path: `${results_root}/${arch}/training/.../weights`

**use the same configuration file `settings/hyperparams.json` for both trainings**


### Experiments
`python experiments/experiments.py --config settings/experiments.json`

Results will be saved in the directory: 

`${results_root}/vae/atlas_experiments[_atlas_net_tn][_edge_length_regularization|_regularize_normal_deviations]/uniform*/${dataset}/${classes}`

HyperCloud model weights are loaded from path: `${results_root}/${arch}/training/.../weights`

LoCondA model weights are loaded from path: `${results_root}/${arch}/atlas_training.../weights`

(make sure that `target_network_input`, `classes`, `use_AtlasNet_TN`, `regularization`, `model` are the same in the `hyperparams.json`/`experiments.json`)


### Configuration settings/experiments.json


##### sphere_triangles_points
Provide input of the HyperCloud Target Network as samples from a triangulation of a unit 3D sphere.

3D points are sampled uniformly from the triangulation of the unit 3D sphere.

Available methods: `hybrid | hybrid2 | hybrid3 | midpoint | midpoint2 | centroid | edge`. 

If enabled, `image_points` will be replaced.


##### square_mesh
Provide input of the LoCondA Target Network as samples from a triangulation of a regular grid (square).

The patch and its edges will be generated from triangulation of the square (square mesh).

If enabled, `patch_points` will be replaced with `grain ^ 2`.

Experiments return the following files for each point cloud:
- `triangulation.pickle` - triangulation of the unit 3D sphere if `sphere_triangles_points` is enabled
- `real.npy` - original point cloud
- `reconstructed.npy` - reconstructed point cloud by HyperCloud Target Network
- `atlas.npy` - patches `image_points x patch_points x 3` 
- `atlas_triangulation.npy` - connections of a patch vertices if `square_mesh` is enabled


### Testing


#### Compute Metrics - JSD, MMD, COV (as in PointFlow)
`python experiments/compute_metrics.py --config settings/experiments.json`


#### Compute Mesh Metrics - Watertightness
```
python experiments/compute_mesh_metrics.py
Arguments:
--shapenet SHAPENET     Path to the directory with ShapeNetCore.v2 meshes
--to_compare TO_COMPARE [TO_COMPARE ...]
                        Multiple paths to different folders where
                        reconstructed meshes are located
--classes CLASSES [CLASSES ...]
                        Classes to be used to compare

Example of usage: python experiments/compute_mesh_metrics.py --shapenet /Datasets/ShapeNetCore.v2/ --to_compare /reconstructions/meshes/ --classes airplane car chair
```
