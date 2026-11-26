# TCHG

Tri-Trust Conditioned Heterogeneous Graph Learning for Dynamic Social Trust Prediction.

## Setup

```bash
conda env create -f environment.yml
conda activate tchg
python -m pip install -e .
```

## Run

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/ciao_70_15_15_static_refined.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/ciaodvd_70_15_15_diffusion.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_robustness.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda --scenarios trust_oriented gnn_oriented
```

Datasets are included in `data/`.

CAT-style robustness perturbs the observed training/validation history at inference with fixed model weights and test candidates.
