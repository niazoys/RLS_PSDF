# Uncertainty-Aware Retinal Layer Segmentation in OCT

This repository contains the official implementation of **uncertainty-aware retinal layer segmentation in optical coherence tomography (OCT)** using **probabilistic signed distance functions (PSDFs)**.

The project was presented at **MIDL 2024** and provides PyTorch-based training and inference code for predicting retinal layer boundaries together with uncertainty estimates.

## Highlights

- Retinal layer segmentation for OCT images.
- Probabilistic signed-distance-function modeling.
- Predictive mean and variance through Gaussian model outputs.
- Optional Monte Carlo dropout and artifact/noise evaluation utilities.
- Hydra-based experiment and configuration management.
- Single-device and distributed GPU training with PyTorch DistributedDataParallel.
- Weights & Biases logging through the project writer utilities.

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
├── environment.yml   # Conda environment specification
└── requirements.txt  # Python package requirements
```

## Installation

The code is intended for Linux environments with Python 3.8 and PyTorch. A CUDA-capable GPU is recommended for training.

### Conda

```bash
git clone https://github.com/niazoys/RLS_PSDF.git
cd RLS_PSDF

conda env create -f environment.yml
conda activate pytorch-template
```

Alternatively, install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

> The pinned environment uses PyTorch 1.13.0 and torchvision 0.14.0. Select a compatible CUDA runtime for your system if you are installing PyTorch separately.

## Data and configuration

The repository does not include OCT data. Prepare the dataset separately and update the data paths in the Hydra configuration before running an experiment.

The default configuration is in [`config/default.yaml`](config/default.yaml). Dataset-specific settings are organized under [`config/datamodule`](config/datamodule), while model and experiment settings are available under the corresponding `config/` subdirectories.

At minimum, set the configured data root to the location of your prepared dataset:

```yaml
data:
  data_root_dir: /path/to/your/data
```

Review the selected datamodule and experiment configuration before starting a run, especially the train, validation, and test splits and the number of retinal layers (`num_class`).

## Training

Training is launched through `trainer.py` and uses Hydra configuration overrides. To run with the default configuration:

```bash
python trainer.py
```

To select an experiment configuration, pass it as a positional argument:

```bash
python trainer.py experiment=<experiment_name>
```

For example, if an experiment is available at `config/experiment/example.yaml`:

```bash
python trainer.py experiment=example
```

Common Hydra overrides can be supplied from the command line, for example:

```bash
python trainer.py \
  device=cuda \
  data.data_root_dir=/path/to/your/data \
  train.num_epoch=100
```

The trainer supports CPU execution, single-GPU execution, and distributed execution based on the configured device and GPU count. Checkpoints and run outputs are written beneath the configured Hydra output directory.

## Inference

Inference is launched through `inference.py` and loads a checkpoint specified by the inference configuration or by the corresponding model-loading settings:

```bash
python inference.py experiment=<experiment_name>
```

Before running inference, verify the following settings:

- The test data directory is configured correctly.
- The model architecture matches the checkpoint.
- `load.resume_state_path` or `load.network_chkpt_path` points to the checkpoint to evaluate.
- `inference_mode` is enabled for the selected configuration.

The inference entry point can run on CPU or GPU and supports distributed execution through the same Hydra configuration system used for training.

## Experiment tracking

Training and evaluation runs use the project writer utilities and can log to Weights & Biases. Configure the relevant logging settings under `config/` and authenticate with Weights & Biases when required:

```bash
wandb login
```

## Reproducibility

Random seeds, model settings, loss functions, data paths, and distributed-training options are controlled through Hydra configuration. Record the selected configuration and checkpoint with each experiment to reproduce results.

## Citation

If you use this code or the PSDF approach in your research, please cite the associated MIDL 2024 paper:

```bibtex
@inproceedings{rls_psdf_midl2024,
  title     = {Uncertainty-aware retinal layer segmentation in OCT through probabilistic signed distance functions},
  booktitle = {Medical Imaging with Deep Learning},
  year      = {2024}
}
```

## License

No license file is currently included in this repository. Please contact the repository authors for permission and licensing information before redistributing the code or using it in a commercial product.
