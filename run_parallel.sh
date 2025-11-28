#!/bin/bash
# TokAlign Parallel Evaluation Script
# Runs baseline (GPU 0) and adapted (GPU 1) evaluations simultaneously

set -e
cd "$(dirname "$0")"

echo "=================================================="
echo "TokAlign Parallel Evaluation - 2x H100 GPUs"
echo "=================================================="
echo ""
echo "GPU 0: Baseline evaluation (EleutherAI/pythia-1b)"
echo "GPU 1: Adapted evaluation (checkpoint-2500)"
echo ""

# Create logs directory
mkdir -p logs

# Step 1: Run tokenizer efficiency (no GPU needed, runs once)
echo "[1/3] Running tokenizer efficiency analysis..."
jupyter nbconvert --to notebook --execute eval_tokenizer_efficiency.ipynb \
    --output eval_tokenizer_efficiency_executed.ipynb \
    2>&1 | tee logs/tokenizer_efficiency.log
echo "✓ Tokenizer analysis complete"
echo ""

# Step 2: Run baseline and adapted in parallel
echo "[2/3] Starting parallel model evaluations..."
echo ""

# Run baseline on GPU 0 in background
echo "  → GPU 0: Starting baseline evaluation..."
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute eval_baseline.ipynb \
    --output eval_baseline_executed.ipynb \
    2>&1 | tee logs/baseline.log &
BASELINE_PID=$!

# Run adapted on GPU 1 in background
echo "  → GPU 1: Starting adapted evaluation..."
CUDA_VISIBLE_DEVICES=1 jupyter nbconvert --to notebook --execute eval_adapted.ipynb \
    --output eval_adapted_executed.ipynb \
    2>&1 | tee logs/adapted.log &
ADAPTED_PID=$!

echo ""
echo "Baseline PID: $BASELINE_PID (GPU 0)"
echo "Adapted PID:  $ADAPTED_PID (GPU 1)"
echo ""
echo "Waiting for both evaluations to complete..."
echo "(Monitor with: tail -f logs/baseline.log logs/adapted.log)"
echo ""

# Wait for both to complete
wait $BASELINE_PID
BASELINE_STATUS=$?
echo "✓ Baseline evaluation complete (exit: $BASELINE_STATUS)"

wait $ADAPTED_PID
ADAPTED_STATUS=$?
echo "✓ Adapted evaluation complete (exit: $ADAPTED_STATUS)"

echo ""
echo "[3/3] Generating summary..."
echo ""

# Summary
echo "=================================================="
echo "EVALUATION COMPLETE"
echo "=================================================="
echo ""
echo "Results saved to: $(pwd)/results/"
ls -la results/
echo ""
echo "Executed notebooks:"
ls -la *_executed.ipynb 2>/dev/null || echo "  (check for errors in logs/)"
echo ""
echo "Logs: $(pwd)/logs/"

