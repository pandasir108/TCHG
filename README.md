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
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciao_70_15_15_static_refined.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciaodvd_70_15_15_diffusion.yaml --device cuda
```

Datasets are included in `data/`. Checkpoints and metrics are saved to `outputs/`.

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests -q
```
