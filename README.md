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
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/epinions_80_10_10_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/ciao_70_15_15_static_refined.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config configs/ciaodvd_70_15_15_diffusion.yaml --device cuda
python runners/generate_poisoning.py --config configs/epinions_70_15_15_memory_days.yaml --output outputs/attacks/trust70
python runners/run_poisoning.py --config configs/epinions_70_15_15_memory_days.yaml --injected-edges outputs/attacks/trust70/rate_0.2/injected_edges.csv --attack-metadata outputs/attacks/trust70/rate_0.2/attack_generation.json --output outputs/poisoning/trust70_20 --prepare-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_multiseed.py --config outputs/poisoning/trust70_20/resolved_poisoning_config.json --device cuda
```

Datasets are included in `data/`.

Poisoning injects edges into training/validation and retrains the model. Test candidates and user groups stay fixed. Epinions adaptations are recorded in the generated protocol files.

T-Spear requires its [upstream code](https://github.com/wooner49/T-spear-shield) and a compiled [TGL sampler](https://github.com/amazon-science/tgl) in the attack Python environment:

```bash
python runners/prepare_tspear.py --upstream ../T-spear-shield --output outputs/attacks/tspear70
python runners/train_tspear_surrogate.py --events data/processed/epinions_full_cat_rolling/events.npz --sampler-root ../tgl --output outputs/attacks/tspear70
python runners/generate_tspear.py --sampler-root ../tgl --output outputs/attacks/tspear70
```

Fixed-weight history perturbations remain available through `runners/run_robustness.py`.
