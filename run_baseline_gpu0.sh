#!/bin/bash
# Run baseline evaluation on GPU 0
cd "$(dirname "$0")"
echo "Running baseline evaluation on GPU 0..."
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute eval_baseline.ipynb \
    --output eval_baseline_executed.ipynb
echo "Done! Results in results/"

