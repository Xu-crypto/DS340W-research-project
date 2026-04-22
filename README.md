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

**Python 3.12 is required.** Python 3.13+ and 3.14 do not work because `torch_scatter` cannot build on them.


##First-Time Setup 

```bash
git clone https://github.com/Xu-crypto/DS340W-research-project

```
### Mac (CPU only)

```bash
cd DS340W-research-project
conda create -n pm25gnn python=3.12 -y
conda activate pm25gnn
pip install -r requirements_mac.txt
```

Find your computer name

The code uses your computer name to select paths.

Run:

Mac/Linux
```bash
python -c "import os; print(os.uname().nodename)"
```
Windows
```bash
python -c "import platform; print(platform.node())"
```
Update config_enhanced.yaml

Open config_enhanced.yaml and edit the filepath: section.
Add your computer name and correct paths:

```bash
filepath:
  Your-Computer-Name:
    knowair_fp: /full/path/to/your/project/data/KnowAir.npy
    results_dir: /full/path/to/your/project/results

```

Run with code:
```bash
python train_enhanced.py
```

### Windows with GPU (CUDA)

```bash
cd DS340W-research-project
conda create -n pm25gnn python=3.12 -y
conda activate pm25gnn

# Install normal packages first
pip install -r requirements_windows.txt

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch Geometric dependency and package
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
pip install torch-geometric
```

Run with:
```bash
python train_enhanced_windows.py
```

Windows uses separate files because `os.uname()` does not exist on Windows:
- `util_windows_fixed.py` — config loader with `platform.node()` fallback
- `dataset_windows.py` — imports from `util_windows_fixed`
- `train_enhanced_windows.py` — imports from the Windows-compatible files

### Notes

- **numpy must stay below 2.0.** Newer numpy breaks scipy and pandas compatibility.
- The code auto-detects GPU. If CUDA is available it uses GPU, otherwise CPU.
- If `torch_scatter` fails to install, try:
  ```bash
  pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
  ```
  Replace `2.10.0` with your torch version (`python -c "import torch; print(torch.__version__)"`)

## Project Structure

```
PM2.5-GNN/
├── data/
│   ├── KnowAir.npy                   # Dataset (download separately)
│   ├── altitude.npy                   # Elevation data
│   └── city.txt                       # City coordinates
├── model/
│   ├── PM25_GNN.py                    # Original model
│   ├── PM25_GNN_Enhanced.py           # Enhanced model (mine)
│   ├── GC_LSTM.py                     # Baseline
│   ├── GRU.py                         # Baseline
│   ├── LSTM.py                        # Baseline
│   ├── MLP.py                         # Baseline
│   └── cells.py                       # GRU/LSTM cells
├── losses.py                          # Frequency-weighted MAE loss
├── graph.py                           # Graph construction
│
├── # --- Mac/Linux ---
├── util.py                            # Config loader
├── dataset.py                         # Data loading
├── train.py                           # Original training script
├── train_enhanced.py                  # Enhanced training script
├── config_enhanced.yaml               # Enhanced config
├── requirements_mac.txt               # Mac dependencies
│
├── # --- Windows ---
├── util_windows_fixed.py              # Config loader (Windows)
├── dataset_windows.py                 # Data loading (Windows)
├── train_enhanced_windows.py          # Enhanced training script (Windows)
├── requirements_windows.txt           # Windows dependencies
│
├── config.yaml                        # Shared config
├── run_experiments.sh                 # Batch runner (Mac/Linux)
└── results/                           # Saved results
```

## Configuration

Edit `config_enhanced.yaml` (Mac) or `config.yaml` (Windows):

```yaml
# Dataset split
dataset_num: 1    # 1 = full year (2yr train), 2 = winter only, 3 = fall/winter

# Model
model: PM25_GNN_Enhanced    # mine
# model: PM25_GNN           # original
# model: GC_LSTM            # baseline
# model: GRU                # baseline

# Loss
loss: Combined    # 0.5*MSE + 0.5*fMAE
# loss: MSE       # original
# loss: fMAE      # frequency-weighted MAE only
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

### Feature Importance (gate weights, dataset 3)

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
- `predict.npy` / `label.npy` — Predictions and ground truth
- `feature_importance.npy` — Gate weight averages per feature
- `gate_weights.npy` — Raw gate weights for all test samples
- `uncertainty.npy` — MC Dropout standard deviations

## References

- Wang et al. (2020) PM2.5-GNN, ACM SIGSPATIAL
- Nedungadi et al. (2025) AirCast, arXiv
- Kalaiselvi et al. (2026) MAST-Net, Scientific Reports
- Raza and Singh (2025) AI-Driven Data Fusion, JSIAR
- Cui et al. (2019) Class-Balanced Loss, CVPR
