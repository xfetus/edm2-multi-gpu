# EDM2 with multi-gpu: Analyzing and Improving the Training Dynamics of Diffusion Models

This is a multi-gpu PyTorch implementation of the paper [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696): 
* This repo only contains configs and experiments on small or medium scale datasets such as CIFAR-10/100 and Tiny-ImageNet. Full re-implementation on ImageNet-1k would be extremely expensive.
* This repo contains implementations of `Config C`, `Config E` and the final `Config G` models. You can compare `block[C/E/G].py` and `unet[C/E/G].py` against each other to learn about the improvements proposed by the authors.

## Installing dependencies using `uv`
```
#install uv (linux-based)
curl -LsSf https://astral.sh/uv/install.sh | sh
#Create a new Python project. `uv init` (run one time to generate pyproject.toml)
#setting env
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e "."
uv pip list --verbose
```

## Usage
Use 4 GPUs to train unconditional Config C/E/G models on the CIFAR-100 dataset:
```bash
GPUS=1
torchrun --nproc_per_node=${GPUS} train.py --config ../../config/cifar100/C.yaml --use_amp
torchrun --nproc_per_node=${GPUS} train.py --config ../../config/cifar100/E.yaml --use_amp
```

To generate 50000 images with different checkpoints, for example, run:
```bash
torchrun --nproc_per_node=4
  sample.py --config config/cifar100/C.yaml --use_amp --epoch 1000
  sample.py --config config/cifar100/E.yaml --use_amp --epoch 1600
  sample.py --config config/cifar100/G.yaml --use_amp --epoch 1999
```

## Observations and takeaways
- Config C shows **consistent improvements** over the original EDM. The main contribution is from the multi-task weighting.
- Config E could perform better than Config C, but the convergence is **significantly slower** (see the epoch counts below).
- Config G seems to **favor latent-space modeling**, and it's inferior to Config E on pixel-space generation.
- Post-hoc EMA (and the power function EMA) tends to **favor longer training** durations. For small-sized datasets like CIFAR, it won't help much.

## Results
We report Config C and Config E results on CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

|        Config       |     Model     | Network size | Best FID (18 steps)   | Best linear probe acc. |
|:--------------------|:--------------|:-------------|:----------------------|:-----------------------|
| cifar10/C.yaml      | Uncond. EDM2C | 39.5M        | 3.03 @ epoch 1000     | 91.85 @ epoch 500      |
| cifar10/E.yaml      | Uncond. EDM2E | 39.5M        | 2.72 @ epoch 2000     | 93.46 @ epoch 1000     |
| cifar100/C.yaml     | Uncond. EDM2C | 39.5M        | 5.06 @ epoch 1000     | 65.40 @ epoch 500      |
| cifar100/E.yaml     | Uncond. EDM2E | 39.5M        | 4.33 @ epoch 2000     | 69.04 @ epoch 1100     |
| tinyimagenet/C.yaml | Uncond. EDM2C | 62.4M        | 15.96 @ epoch 1600*   | 50.99 @ epoch 600      |
| tinyimagenet/E.yaml | Uncond. EDM2E | 62.4M        | 16.79 @ epoch 1500*   | 52.07 @ epoch 1400     |

*Note: Unfinished training (due to high computational cost). The FID has not saturated, and keep training can lead to lower FIDs.


## Citations

```bibtex
@article{karras2023analyzing,
  title={Analyzing and Improving the Training Dynamics of Diffusion Models},
  author={Karras, Tero and Aittala, Miika and Lehtinen, Jaakko and Hellsten, Janne and Aila, Timo and Laine, Samuli},
  journal={arXiv preprint arXiv:2312.02696},
  year={2023}
}
```

