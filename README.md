# PM2.5-GNN Enhanced

**Attention-Enhanced Graph Neural Network with Frequency-Weighted Loss for PM2.5 Forecasting**

Xu Wang | College of Engineering, Penn State | xjw5184@psu.edu

DS340W Research Project

## Overview

This project builds on [PM2.5-GNN](https://github.com/shawnwang-tech/PM2.5-GNN) (Wang et al., SIGSPATIAL 2020) with three modifications:

1. **Multi-head attention** in the GNN message passing so the model learns which neighboring cities matter more
2. **Dynamic feature selection gate** that picks which weather features to focus on at each timestep
3. **Frequency-weighted MAE loss** that puts more weight on rare high-pollution samples

The model also includes MC Dropout uncertainty estimation and outputs interpretable feature importance rankings.

## Dataset

**KnowAir** — 184 Chinese cities, 2015-2018, 3-hour intervals

Download from [Google Drive](https://drive.google.com/open?id=1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L) and place `KnowAir.npy` in the `data/` folder.

## Setup

### Option 1: Conda (Recommended)

```bash
conda create -n pm25gnn python=3.12 -y
conda activate pm25gnn
pip install -r requirements.txt
```

### Option 2: venv

```bash
python3.12 -m venv pm25env
source pm25env/bin/activate        # Mac/Linux
# pm25env\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
```

### Windows with GPU (CUDA)

If you have an NVIDIA GPU, install PyTorch with CUDA first, then the rest:

```bash
conda create -n pm25gnn python=3.12 -y
conda activate pm25gnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_geometric
pip install -r requirements.txt
```

### Notes

- **Python 3.12 is required.** Python 3.13+ and 3.14 do not work because `torch_scatter` cannot build on them.
- **numpy must stay below 2.0.** Newer numpy breaks scipy and pandas compatibility.
- If `torch_scatter` fails to install, try:
  ```bash
  pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
  ```
  Replace `2.10.0` with your torch version (`python -c "import torch; print(torch.__version__)"`)

## Project Structure

```
PM2.5-GNN/
├── data/
│   ├── KnowAir.npy              # Dataset (download separately)
│   ├── altitude.npy              # Elevation data for graph construction
│   └── city.txt                  # City coordinates
├── model/
│   ├── PM25_GNN.py               # Original model
│   ├── PM25_GNN_Enhanced.py      # Enhanced model (mine)
│   ├── GC_LSTM.py                # Baseline
│   ├── GRU.py                    # Baseline
│   ├── LSTM.py                   # Baseline
│   ├── MLP.py                    # Baseline
│   └── cells.py                  # GRU/LSTM cell implementations
├── losses.py                     # Frequency-weighted MAE loss
├── train.py                      # Original training script
├── train_enhanced.py             # Enhanced training script (mine)
├── config.yaml                   # Original config
├── config_enhanced.yaml          # Enhanced config (mine)
├── graph.py                      # Graph construction
├── dataset.py                    # Data loading
├── util.py                       # Config loader
├── run_experiments.sh             # Batch experiment runner
├── requirements.txt              # Dependencies
└── results/                      # Saved results
```

## Running

### Quick Start

```bash
python train_enhanced.py
```

This runs the Enhanced model with Combined loss on dataset 3 by default.

### Change Dataset

Edit `config_enhanced.yaml`:

```yaml
dataset_num: 1    # 1 = full year (2yr train), 2 = winter only, 3 = fall/winter (smallest)
```

### Change Model

```yaml
model: PM25_GNN_Enhanced    # mine
# model: PM25_GNN           # original
# model: GC_LSTM            # baseline
# model: GRU                # baseline
```

### Change Loss

```yaml
loss: Combined    # 0.5*MSE + 0.5*fMAE (mine)
# loss: MSE       # original
# loss: fMAE      # frequency-weighted MAE only
```

### Run All Experiments

```bash
chmod +x run_experiments.sh
./run_experiments.sh ablation
```

## Results

Tested on all 3 sub-datasets, 10 runs each:

| Model | Dataset | RMSE | MAE | R² | CSI% |
|---|---|---|---|---|---|
| GRU | 1 | 21.00 | - | - | 45.38 |
| GC-LSTM | 1 | 20.84 | - | - | 45.83 |
| PM2.5-GNN | 1 | 19.93 | - | - | 48.52 |
| **Enhanced** | **1** | **20.31** | **15.99** | **0.556** | **45.27** |
| GRU | 2 | 32.59 | - | - | 51.07 |
| PM2.5-GNN | 2 | 31.37 | - | - | 52.33 |
| **Enhanced** | **2** | **31.91** | **25.32** | **0.479** | **49.90** |
| GRU | 3 | 45.25 | - | - | 59.40 |
| PM2.5-GNN | 3 | 43.29 | - | - | 61.91 |
| **Enhanced** | **3** | **43.88** | **35.98** | **0.494** | **59.77** |

RMSE in µg/m³. Baseline numbers from Wang et al. (2020).

### Feature Importance (from gate weights, dataset 3)

| Rank | Feature | Weight |
|---|---|---|
| 1 | Surface Pressure | 0.783 |
| 2 | 2m Temperature | 0.728 |
| 3 | Hour of Day | 0.699 |
| 4 | U-Wind (950 hPa) | 0.690 |
| 5 | Boundary Layer Height | 0.631 |
| ... | ... | ... |
| 13 | PM2.5 History | 0.481 |

### Uncertainty

MC Dropout (20 samples): mean std = 11.63 ± 0.25 µg/m³

## Output Files

Results saved to `results/1_24/{dataset}/{model_name}/`:

- `metric.txt` — RMSE, MAE, R², CSI, POD, FAR (mean ± std)
- `predict.npy` — Predicted PM2.5 values
- `label.npy` — Ground truth PM2.5 values
- `feature_importance.npy` — Gate weight averages per feature
- `gate_weights.npy` — Raw gate weights for all test samples
- `uncertainty.npy` — MC Dropout standard deviations

## References

- Wang et al. (2020) PM2.5-GNN, ACM SIGSPATIAL
- Nedungadi et al. (2025) AirCast, arXiv
- Kalaiselvi et al. (2026) MAST-Net, Scientific Reports
- Raza and Singh (2025) AI-Driven Data Fusion, JSIAR
- Cui et al. (2019) Class-Balanced Loss, CVPR
