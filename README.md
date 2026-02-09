# VOADT: Vectorized Trajectory Learning & Online Anomaly Detection

This repository contains two related pipelines:

- **VecLSTM**: trajectory vectorization + CNN/LSTM modeling for efficient trajectory learning (e.g., activity recognition / classification), with optional metadata support.
- **VOADT / VOADT-Distill**: vectorized online anomaly detection for trajectories, including a **distilled student** that mimics a strong multi-scale teacher (MST-OATD) for fast inference.

Most experimental outputs (plots, tables, runtime JSONs) are saved under `paper/`.


## Overview

### VecLSTM (vectorization + efficient sequence learning)
VecLSTM addresses scalability and compute bottlenecks in processing variable-length trajectories:
- **Vectorization**: transforms raw trajectory points into fixed-shape feature grids/tensors for efficient batching.
- **CNN + LSTM**: CNN extracts local spatial/feature patterns; LSTM/GRU captures temporal structure.
- **Metadata**: optional metadata (time, location, labels, user info) to support downstream tasks.

### VOADT (vectorized online anomaly detection for trajectories)
VOADT uses **dynamic vectorization** to transform variable-length trajectories into fixed-shape **feature--time tensors** and performs efficient scoring:
- **VOADT (VecLSTM-style)**: reconstruction-based and/or embedding + GMM scoring on vectorized sequences (see `VOADT/voadt_veclstm_gmm.py`).
- **VOADT-Distill**: a lightweight **Transformer** student distilled from MST-OATD using rank-normalized teacher targets + rank-preserving loss (see `VOADT/voadt_distill_mstoatd.py`).

## Datasets

### GeoLife Dataset
The GeoLife dataset contains GPS trajectories from 182 users over three years, providing a comprehensive view of outdoor activities. It includes 1,467,652 samples with 7 distinct labels. This dataset is ideal for training models that predict human activity from trajectory data.

- **Dataset details**: [GeoLife Dataset](https://www.microsoft.com/en-us/research/project/geolife/)
- **Usage**: The GeoLife dataset is used to evaluate the VecLSTM model on large-scale real-world trajectory data.

## Methodology

VecLSTM follows these key steps:

1. **Data Preprocessing**: Raw GPS data is cleaned, normalized, and prepared for model training.
2. **Vectorization**: A vectorization function transforms trajectory data into a 10x10 grid, storing spatial and temporal features.
3. **CNN and LSTM Model**: A CNN-based model extracts spatial features, and an LSTM captures temporal dependencies from the vectorized data.
4. **Model Training**: The model is trained on the preprocessed and vectorized trajectory data to predict future trajectory points or activity labels.


## Installation

To install and run the VecLSTM model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/solmazsm/VecLSTM.git
    cd VecLSTM
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the model:
    ```bash
    python train_model.py
    ```

## VOADT quickstart (WSL recommended)

Most VOADT experiments in this repo are run in WSL (Ubuntu) with conda (example env name: `voadt_cuda`):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voadt_cuda
```

### Generate outlier test sets (Detour / Route-switching)
Outlier files are stored under `MST-OATD-baseline/data/<dataset>/` as:
- `outliers_data_init_{d}_{alpha}_{rho}.npy` and `outliers_idx_init_{d}_{alpha}_{rho}.npy`
- `outliers_data_rs_{beta}_{rho}.npy` and `outliers_idx_rs_{beta}_{rho}.npy`

Example (Detour):

```bash
cd MST-OATD-baseline
python generate_outliers.py --dataset porto --mode d --distance 2 --fraction 0.2 --obeserved_ratio 1.0
cd ..
```

### Train / evaluate VOADT-Distill (student)

```bash
python VOADT/voadt_distill_mstoatd.py \
  --dataset porto --device cuda \
  --distance 2 --fraction 0.2 --observed_ratio 1.0 \
  --epochs 60 --batch_size 512 \
  --save_student paper/voadt_student_porto.pth
```

Saved scores will appear under `VOADT/probs/` (e.g., `voadt_distill_scores_*.npy`).

### VOADT (VecLSTM-style) recon/GMM scoring

```bash
python VOADT/voadt_veclstm_gmm.py \
  --dataset porto --device cuda \
  --distance 2 --fraction 0.2 --observed_ratio 1.0 \
  --num_bins 16,32,64 --ae_epochs 60 --gmm_components 20 --score_mode both
```


## Datasets (VOADT / MST-OATD)

VOADT experiments in this repo use three city-scale taxi trajectory datasets prepared in the **MST-OATD format** (tokenized spatial grid + time vectors). 


### Expected files (per dataset)

- **Normal splits**
  - `train_data_init.npy`
  - `test_data_init.npy`
- **Synthetic outliers (Detour)**
  - `outliers_data_init_{d}_{alpha}_{rho}.npy`
  - `outliers_idx_init_{d}_{alpha}_{rho}.npy`

### Beijing metadata (`bj/meta.json`)

Beijing (T-Drive) also includes:
- `MST-OATD-baseline/data/bj/meta.json`

This file stores the grid boundary, grid size, and time interval used for tokenization.

### Note on raw data

This repository contains some raw data under `data/` and `datasets/`, but **VOADT/MST-OATD experiments use the processed `.npy` files** under `MST-OATD-baseline/data/`. 

```bash
cd MST-OATD-baseline
python generate_outliers.py --dataset <porto|cd|bj> --mode <d|rs> ...
cd ..
```

### Train / evaluate VOADT-Distill (student)

### VOADT (VecLSTM-style) recon/GMM scoring

### Case-study figures (normal vs outlier overlays)


### Detection-time benchmarking (ms/trajectory)

```bash
# VOADT-Distill (student inference)
python VOADT/tools/benchmark_voadt_detection_time.py --student_ckpt paper/voadt_student_porto.pth --dataset porto --device cuda --out_json paper/runtime_voadt_porto.json

# MST-OATD
python VOADT/tools/benchmark_mstoatd_detection_time.py --dataset porto --device cuda --out_json paper/runtime_mstoatd_porto.json

# VOADT VecLSTM-style scoring (recon/gmm/both)
python VOADT/tools/benchmark_voadt_veclstm_scoring_time.py --dataset porto --mode both --device cuda --out_json paper/runtime_voadt_veclstm_porto_both.json
```

### PR-AUC vs detection-time scatter

### Detection-time benchmarking (ms/trajectory)

```bash
# VOADT-Distill (student inference)

# MST-OATD

# VOADT VecLSTM-style scoring (recon/gmm/both)

```


# Project Structure

The following is the structure of this project:

```
VecLSTM/
├── VecLSTM/                         # VecLSTM core code (vectorization + CNN/LSTM)
├── vectorization/                   # vectorization utilities
├── datasets/                        # datasets & dataset notes
├── MST-OATD-baseline/               # teacher/baseline (MST-OATD) + generated outlier files
│   ├── data/{porto,cd,bj}/
│   ├── probs/                       # saved MST-OATD scores
│   └── models/                      # pretrained teacher checkpoints
├── VOADT/                           # VOADT / VOADT-Distill code + score outputs
│   ├── probs/                       # saved VOADT/VOADT-Distill score files
│   └── tools/                       # plotting, benchmarking, automation scripts
├── paper/                           # figures/tables/JSON benchmarks 
├── Figures/                         
├── models/                         
├── LICENSE
└── README.md
```
  

For any questions, concerns, or comments for improvements, etc, please create an issue on the issues page for this project, or email the authors directly.


