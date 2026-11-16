# TCHG Data

This directory contains the public trust datasets and processed files used by TCHG experiments.

## Included public datasets

- `Epinions`
- `Ciao`
- `CiaoDVD`

## Directory layout

- `raw/`: original public dataset files
- `processed/`: processed event tensors and split metadata used by experiments

## Raw data

- `raw/epinions/`
  - `user_user_time.txt`
  - `user_item_time.txt`
- `raw/ciao/`
  - `trust.txt`
  - `rating_with_timestamp.txt`
  - `catalog_ciao.txt`
  - `ciao.zip`
  - `ciao_with_rating_timestamp_txt.zip`
- `raw/ciaodvd/`
  - `CiaoDVD.zip`

## Processed data

- `processed/epinions/`: base processed Epinions data
- `processed/epinions_full/`: full processed Epinions data
- `processed/epinions_full_strict/`: strict processed Epinions split
- `processed/epinions_full_cat_rolling/`: 70-15-15 rolling split files used in experiments
- `processed/epinions_full_cat_rolling_80_10_10/`: 80-10-10 rolling split files
- `processed/ciao_full_cat_rolling/`: processed Ciao rolling split files
- `processed/ciao_full_cat_rolling_80_10_10/`: processed Ciao 80-10-10 rolling split files
- `processed/ciao_full_cat_rolling_node_onehot/`: processed Ciao variant with one-hot node features
- `processed/ciao_full_cat_rolling_train_struct/`: processed Ciao train-structure variant
- `processed/ciaodvd_full_cat_rolling/`: processed CiaoDVD rolling split files
- `processed/ciaodvd_full_cat_rolling_80_10_10/`: processed CiaoDVD 80-10-10 rolling split files

No CAT-specific source package is included here.
