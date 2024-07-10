# Learning Dynamic Neural Radiance Fields with Smooth Deformation Priors
[Thesis](https://drive.google.com/drive/u/0/folders/1r1MpzEuJe07S-EsKOk7TBWWXw9d9112J)
## Abstract
With the advent of neural radiance fields for novel view synthesis, new methods are emerging to generalize these models to dynamic data, e.g., multi-view videos. Modelling 3D motion via 2D observations is a non-trivial problem, which has resulted in limitations for previous proposals, especially concerning the temporal coherence of the output learned scene. In this thesis work, we present a technique for learning dynamic scenes using Lipschitz regularization. Specifically, time is considered as an additional input to static neural radiance fields and the motion is modelled as a temporal distortion function, implemented as an additional neural network, which is applied to a canonical space. The Lipschitz regularization is applied to this temporal deformation, allowing for smooth dynamics while the canonical space can learn geometry and colour information with arbitrarily high frequency. Both mappings are implemented as MLPs. In our evaluation, we tested the effectiveness of Lipschitz regularization on scenes with rigid, non-rigid and articulated objects with non-Lambertian materials and on multiple neural radiance field architectures. Our experiments show that applying Lipschitz regularization on temporal distortion enables dynamic radiance fields to learn a smooth dynamic scene with improved temporal coherence and fidelity.
## Workflow
<img src="https://github.com/lorenzo-delsignore/smooth-dnerf/assets/66021430/a5ffd36c-9269-4fc1-9c62-e7b63141ce07" alt="Workflow" width="600">


## Installation
1. [Install Nerfstudio](https://docs.nerf.studio/quickstart/installation.html)
2. Clone this repository ```git clone https://github.com/lorenzo-delsignore/smooth-dnerf.git)```
3. Install this repository as a python package ```pip install -e .```
4. Check if the dynamic models ```nerfacto-dnerf```, ```nerfacto-nerfplayer-dnerf``` and ```vanilla-dnerf``` are listed with the command ```ns-train -h```
## Usage
- The only dataset supported is the [DNeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) dataset
- Train a model using the following command: ```ns-train <name model> --data <data path>```
- Model implemented are D-Nerfacto ```nerfacto-dnerf```, D-NerfPlayer-Nerfacto ```nerfacto-nerfplayer-dnerf```, D-NeRF ```vanilla-dnerf```
- Monitor the training with [wandb](https://wandb.ai/).
## Examples
### D-NerfPlayer-Nerfacto Jumping jacks
#### With Lipschitz Regularization

https://github.com/lorenzo-delsignore/smooth-dnerf/assets/66021430/59c8fd7b-a08a-4b26-8e8c-1827471f389a
#### Without Lipschtiz Regularization
https://github.com/lorenzo-delsignore/smooth-dnerf/assets/66021430/34111cc8-d810-4baa-8d22-2596e3bb9eec
### Lego
#### D-NerfPlayer-Nerfacto with Lipschitz Regularization
https://github.com/lorenzo-delsignore/smooth-dnerf/assets/66021430/f189ea2b-7803-4ed3-8cb9-0b53e221fabb
#### Comparision with state of the art models
[NeRFPlayer](https://github.com/lsongx/nerfplayer-nerfstudio)

https://github.com/lorenzo-delsignore/smooth-dnerf/assets/66021430/ecbe47ed-0266-4f34-9c00-eb2d1adf5f92



