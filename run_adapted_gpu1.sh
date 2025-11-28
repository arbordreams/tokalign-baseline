#!/bin/bash
# Run adapted model evaluation on GPU 1
cd "$(dirname "$0")"
echo "Running adapted model evaluation on GPU 1..."
CUDA_VISIBLE_DEVICES=1 jupyter nbconvert --to notebook --execute eval_adapted.ipynb \
    --output eval_adapted_executed.ipynb
echo "Done! Results in results/"

