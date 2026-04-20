# Energy-Weighted Flow Matching

Official code for the paper [Energy-Weighted Flow Matching: Unlocking Continuous Normalizing Flows for Efficient and Scalable Boltzmann Sampling](https://arxiv.org/abs/2509.03726).

This repository implements Energy-Weighted Flow Matching (EWFM) for Boltzmann sampling. The implementation builds on the codebase from the [iDEM-paper](https://github.com/jarridrb/DEM).

Similar to the iDEM codebase, the GMM task uses the [FAB torch](https://github.com/lollcat/fab-torch) code, and the project layout and Hydra/Lightning utilities are based on the [hydra lightning template](https://github.com/ashleve/lightning-hydra-template).

## Structure

- `ewfm/`: Core implementation of the models and methods
- `data/`: Contains the data files (LJ13, LJ55, and DW4 datasets; GMM data is sampled on-the-fly during training)
- `configs/`: Hydra configuration files
- `pipelines/`: SLURM experiment scripts
  - `ewfm/`: Baseline EWFM experiments
  - `i_ewfm/`: Iterative EWFM (iEWFM) experiments
  - `a_ewfm/`: Annealed EWFM (aEWFM) experiments

## System Requirements

Our experiments were run on GPUs (specific hardware details are outlined in the paper). The codebase has not been tested on CPU-only systems.

## Getting Started

**Prerequisites:** CUDA GPU, Conda (plus Mamba if you want to run the SLURM scripts as-is — they call `mamba activate`), W&B account (optional)

**Setup:**

```bash
conda env create -f environment.yaml
conda activate ewfmenv
pip install --no-deps git+https://github.com/noegroup/bgflow.git git+https://github.com/lollcat/fab-torch.git
cp .env.example .env   # edit with your W&B credentials (or skip), then `source .env`
```

> **Note:** The environment installs PyTorch with CUDA 12.1. For a different CUDA version, change `pytorch-cuda=12.1` in `environment.yaml` (e.g., `pytorch-cuda=11.8`). See [PyTorch installation](https://pytorch.org/get-started/locally/) for available options.

**For SLURM scripts:** Edit paths in `.sbatch` files:

- `cd /path/to/your/ewfm/` → your project directory
- `WANDB_API_KEY` and `WANDB_ENTITY` → your W&B credentials
- Uncomment `#SBATCH --partition=` and `#SBATCH --qos=` and set them for your cluster

## Running Experiments

### Train model

To run the default training configuration (defined in `configs/train.yaml` with the `gmm` experiment):

```bash
python ewfm/train.py
```

You can override configuration parameters from the command line. For example, to change the seed:

```bash
python ewfm/train.py seed=42
```

### Full Experimental Pipeline

The complete experimental pipeline consists of 4 steps (as implemented in the SLURM scripts):

1. **Train Initial Model**: Train EWFM model using energy evaluations only
2. **Generate Samples**: Generate samples from trained model for evaluation
3. **Train Evaluation Model**: Train a separate flow matching model on generated samples for fair ESS/NLL evaluation (as done in the iDEM-paper)
4. **Final Evaluation**: Calculate final ESS and NLL metrics using the evaluation model

In the following, we will quickly go through each of these steps and give an example command.

#### Running Individual Steps

**Step 1: Train model**

```bash
python ewfm/train.py experiment=gmm
```

Train the EWFM model using only energy function evaluations on samples from a proposal distribution.

**Step 2: Generate samples**

```bash
python ewfm/eval.py ckpt_path=path/to/checkpoint.ckpt save_samples=true num_samples=100000
```

Generate samples from the trained model to create a dataset for training the evaluation model.

**Step 3: Train evaluation model**

```bash
python ewfm/train.py model.use_train_data=true energy.data_path_train=path/to/samples.npy
```

Train a separate flow matching model on the generated samples to enable fair calculation of ESS and NLL metrics.

**Step 4: Final evaluation (ESS & NLL)**

```bash
python ewfm/eval.py ckpt_path=path/to/eval_model_checkpoint.ckpt
```

Compute the Effective Sample Size (ESS) and Negative Log-Likelihood (NLL) using the evaluation model.

For complete automated pipelines, see the SLURM scripts in `pipelines/`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{dern2025energy,
  title={Energy-Weighted Flow Matching: Unlocking Continuous Normalizing Flows for Efficient and Scalable Boltzmann Sampling},
  author={Dern, Niclas and Redl, Lennart and Pfister, Sebastian and Kollovieh, Marcel and L{\"u}dke, David and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2509.03726},
  year={2025}
}
```
