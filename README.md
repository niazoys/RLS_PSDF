# Uncertainty-aware Retinal Layer Segmentation in OCT through Probabilistic Signed Distance Functions

<p align="center">
  <a href="https://arxiv.org/abs/2412.04935"><img src="https://img.shields.io/badge/arXiv-2412.04935-b31b1b.svg" alt="arXiv"></a>
  <a href="https://proceedings.mlr.press/v250/islam24a.html"><img src="https://img.shields.io/badge/PMLR-MIDL%202024-4b8bbe.svg" alt="PMLR / MIDL 2024"></a>
  <a href="https://niazoys.github.io/RLS_PSDF/"><img src="https://img.shields.io/badge/Project%20Page-Website-2ea44f.svg" alt="Project page"></a>
  <a href="https://github.com/niazoys/RLS_PSDF"><img src="https://img.shields.io/github/stars/niazoys/RLS_PSDF?style=flat" alt="GitHub stars"></a>
  <a href="https://github.com/niazoys/RLS_PSDF"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="GitHub"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2412.04935">Paper</a> &nbsp;|&nbsp;
  <a href="https://proceedings.mlr.press/v250/islam24a.html">MIDL 2024 / PMLR</a> &nbsp;|&nbsp;
  <a href="https://niazoys.github.io/RLS_PSDF/">Project page</a> &nbsp;|&nbsp;
  <a href="https://github.com/niazoys/RLS_PSDF">Code</a>
</p>

This repository contains the official implementation of **uncertainty-aware retinal layer segmentation in optical coherence tomography (OCT)** using **probabilistic signed distance functions (pSDFs)**.

<p align="center">
  <img src="https://raw.githubusercontent.com/niazoys/RLS_PSDF/project-page/static/images/fig1_final-1.png" alt="Probabilistic signed distance functions for retinal layer segmentation" width="900">
</p>

<p align="center"><em>Figure 1 from the paper: signed distance functions represent retinal layer geometry through level sets, while probabilistic modeling provides spatially meaningful uncertainty.</em></p>

## Highlights

- **Geometric segmentation:** retinal layer boundaries are represented by signed distance functions and their level sets.
- **Probabilistic predictions:** the model estimates a mean and variance for the signed distance representation.
- **Uncertainty-aware analysis:** uncertainty can reveal ambiguous, noisy, or pathological regions in OCT scans.
- **Robustness experiments:** the project includes settings for evaluating synthetic artifacts and noise.
- **Hydra configuration:** models, datasets, losses, experiments, and logging are configured with composable YAML files.
- **Scalable training:** CPU, single-GPU, and distributed GPU execution are supported through PyTorch.

## Repository structure

```text
.
├── config/          # Hydra configuration files and experiment settings
├── dataloader/      # DataLoader construction and sampling utilities
├── dataset/         # Dataset definitions and data-related code
├── model/           # Network architectures and model handlers
├── utils/           # Losses, logging, writers, and helper functions
├── trainer.py       # Training entry point
├── inference.py     # Checkpoint-based inference entry point
├── test.py          # Data-loading and evaluation utilities
├── environment.yml  # Conda environment specification
└── requirements.txt # Python package requirements
```

## Installation

The provided environment targets Linux, Python 3.8, PyTorch 1.13, and CUDA-enabled execution. A GPU is recommended for training.

### Conda

```bash
git clone https://github.com/niazoys/RLS_PSDF.git
cd RLS_PSDF

conda env create -f environment.yml
conda activate pytorch-template
```

Alternatively, install the package dependencies with pip:

```bash
pip install -r requirements.txt
```

> If you install PyTorch separately, choose a version and CUDA runtime compatible with your system. The pinned environment specifies `torch==1.13.0` and `torchvision==0.14.0`.

### Docker

A CUDA 11.3-based Dockerfile is included for users who prefer a containerized setup:

```bash
docker build -f DockerFile -t rls-psdf .
docker run --gpus all -it --rm -v "$PWD":/app rls-psdf
```

Check the Dockerfile and host NVIDIA Container Toolkit configuration before launching GPU workloads.

## Data and configuration

The OCT datasets are not included in this repository. Prepare the data separately and update the Hydra configuration with the appropriate data locations and train/validation/test splits.

The main configuration is [`config/default.yaml`](config/default.yaml). Dataset, model, and experiment configurations are organized under the corresponding subdirectories of `config/`.

At minimum, set the data root to your prepared dataset:

```yaml
data:
  data_root_dir: /path/to/your/data
```

Before training or inference, also verify the dataset-specific configuration, the number of output classes/layers, checkpoint paths, and the selected device.

## Training

Run the default experiment with:

```bash
python trainer.py
```

Hydra experiment configurations can be selected with a positional argument:

```bash
python trainer.py experiment=<experiment_name>
```

For example, for `config/experiment/example.yaml`:

```bash
python trainer.py experiment=example
```

Configuration values can be overridden from the command line:

```bash
python trainer.py \
  device=cuda \
  data.data_root_dir=/path/to/your/data \
  train.num_epoch=100
```

The trainer creates Hydra run directories under `outputs/` and can log experiments through Weights & Biases when enabled in the configuration:

```bash
wandb login
```

## Inference

Run inference with the same Hydra configuration system:

```bash
python inference.py experiment=<experiment_name>
```

Before running inference, make sure that:

- the test data directory is configured;
- the model architecture matches the checkpoint;
- `load.resume_state_path` or `load.network_chkpt_path` points to the desired checkpoint; and
- inference mode and output-saving options are enabled in the selected configuration.

The inference entry point supports CPU, single-GPU, and distributed GPU execution according to the active configuration.

## Project page

The project website includes the paper figures, an intuitive explanation of the SDF and uncertainty formulation, qualitative examples, comparisons, and the complete BibTeX entry:

**[https://niazoys.github.io/RLS_PSDF/](https://niazoys.github.io/RLS_PSDF/)**

## Citation

If you use this code or the probabilistic signed distance function approach, please cite:

```bibtex
@InProceedings{pmlr-v250-islam24a,
  title     = {Uncertainty-aware retinal layer segmentation in OCT through probabilistic signed distance functions},
  author    = {Islam, Mohammad Mohaiminul and de Vente, Coen and Liefers, Bart and Klaver, Caroline and Bekkers, Erik J. and S{\'a}nchez, Clara I.},
  booktitle = {Proceedings of The 7th International Conference on Medical Imaging with Deep Learning},
  pages     = {672--693},
  year      = {2024},
  volume    = {250},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v250/islam24a.html}
}
```

## License

No license file is currently included in this repository. Please contact the authors before redistributing the code or using it in a commercial product.
