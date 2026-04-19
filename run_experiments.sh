#!/bin/bash
# ============================================================
# Run script for PM2.5-GNN Enhanced experiments
# Usage:
#   ./run_experiments.sh all        # Run all experiments
#   ./run_experiments.sh baselines  # Run only baselines
#   ./run_experiments.sh enhanced   # Run only enhanced model
#   ./run_experiments.sh ablation   # Run ablation study
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_experiment() {
    local model=$1
    local loss=$2
    local dataset=$3
    local extra_desc=$4

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: model=$model, loss=$loss, dataset=$dataset $extra_desc${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Create a temporary config
    cp config_enhanced.yaml config_temp.yaml

    # Use sed to update model and loss in the temp config
    # First, comment out all model lines, then uncomment the desired one
    if [[ "$model" == "PM25_GNN_Enhanced" ]]; then
        sed -i.bak "s/^  model:.*/  model: PM25_GNN_Enhanced/" config_temp.yaml
    elif [[ "$model" == "PM25_GNN" ]]; then
        sed -i.bak "s/^  model:.*/  model: PM25_GNN/" config_temp.yaml
    elif [[ "$model" == "GC_LSTM" ]]; then
        sed -i.bak "s/^  model:.*/  model: GC_LSTM/" config_temp.yaml
    elif [[ "$model" == "GRU" ]]; then
        sed -i.bak "s/^  model:.*/  model: GRU/" config_temp.yaml
    elif [[ "$model" == "LSTM" ]]; then
        sed -i.bak "s/^  model:.*/  model: LSTM/" config_temp.yaml
    elif [[ "$model" == "MLP" ]]; then
        sed -i.bak "s/^  model:.*/  model: MLP/" config_temp.yaml
    fi

    # Update loss
    sed -i.bak "s/^  loss:.*/  loss: $loss/" config_temp.yaml

    # Update dataset number
    sed -i.bak "s/^  dataset_num:.*/  dataset_num: $dataset/" config_temp.yaml

    # Copy temp config to the name train_enhanced.py expects
    cp config_temp.yaml config_enhanced.yaml

    # Run
    python train_enhanced.py

    # Cleanup
    rm -f config_temp.yaml config_temp.yaml.bak

    echo -e "${YELLOW}Done: model=$model, loss=$loss, dataset=$dataset${NC}"
    echo ""
}

case "${1:-enhanced}" in
    baselines)
        echo "Running baseline models..."
        for ds in 1 2 3; do
            run_experiment "MLP" "MSE" "$ds"
            run_experiment "GRU" "MSE" "$ds"
            run_experiment "LSTM" "MSE" "$ds"
            run_experiment "GC_LSTM" "MSE" "$ds"
            run_experiment "PM25_GNN" "MSE" "$ds"
        done
        ;;

    enhanced)
        echo "Running enhanced model..."
        for ds in 1 2 3; do
            run_experiment "PM25_GNN_Enhanced" "Combined" "$ds"
        done
        ;;

    ablation)
        echo "Running ablation study..."
        # Each row adds one component on top of the previous
        for ds in 1 2 3; do
            # Base: original PM25_GNN with MSE
            run_experiment "PM25_GNN" "MSE" "$ds" "(base)"

            # + fMAE loss only (no architecture change)
            run_experiment "PM25_GNN" "fMAE" "$ds" "(+fMAE)"

            # + Combined loss only (no architecture change)
            run_experiment "PM25_GNN" "Combined" "$ds" "(+Combined)"

            # + Enhanced model with MSE (attention + feature selection, no loss change)
            run_experiment "PM25_GNN_Enhanced" "MSE" "$ds" "(+Attn+DFS)"

            # Full: Enhanced model + Combined loss
            run_experiment "PM25_GNN_Enhanced" "Combined" "$ds" "(Full)"
        done
        ;;

    all)
        echo "Running ALL experiments..."
        for ds in 1 2 3; do
            # Baselines
            run_experiment "MLP" "MSE" "$ds"
            run_experiment "GRU" "MSE" "$ds"
            run_experiment "LSTM" "MSE" "$ds"
            run_experiment "GC_LSTM" "MSE" "$ds"
            run_experiment "PM25_GNN" "MSE" "$ds"

            # Ablation
            run_experiment "PM25_GNN" "fMAE" "$ds"
            run_experiment "PM25_GNN" "Combined" "$ds"
            run_experiment "PM25_GNN_Enhanced" "MSE" "$ds"

            # Full enhanced
            run_experiment "PM25_GNN_Enhanced" "Combined" "$ds"
        done
        ;;

    *)
        echo "Usage: $0 {baselines|enhanced|ablation|all}"
        exit 1
        ;;
esac

echo -e "${GREEN}All experiments complete!${NC}"
