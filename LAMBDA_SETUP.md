# Lambda H100 Setup Guide

Instructions for setting up the TokAlign ACL evaluation framework on a Lambda Labs H100 GPU instance running Ubuntu 22.04 with Lambda Stack.

## Step 1: Create Isolated Workspace

```bash
mkdir -p ~/projects/tokalign-eval
cd ~/projects/tokalign-eval
```

## Step 2: Clone Repository

```bash
git clone https://github.com/arbordreams/tokalign-baseline.git
cd tokalign-baseline
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install jupyterlab nbconvert
```

## Step 4: Configure Model Paths

Edit the notebook configuration cells:

**In `eval_tokenizer_efficiency.ipynb` (Cell 1):**
```python
ADAPTED_MODEL = "<PATH_TO_ADAPTED_MODEL>"  # Update this
```

**In `eval_adapted.ipynb` (Cell 1):**
```python
MODEL_PATH = "<PATH_TO_ADAPTED_MODEL>"  # Update this
```

Replace `<PATH_TO_ADAPTED_MODEL>` with the actual path or Hugging Face model ID for the vocabulary-adapted model being evaluated.

## Step 5: Verify GPU Setup

```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute Capability: {torch.cuda.get_device_capability()}'); print(f'CUDA: {torch.version.cuda}')"
```

**Expected output:**
```
GPU: NVIDIA H100 80GB HBM3
Compute Capability: (9, 0)
CUDA: 12.x
```

## Step 6: Create Results Directory

```bash
mkdir -p results
```

## Directory Structure After Setup

```
~/projects/tokalign-eval/
└── tokalign-baseline/
    ├── results/                       # All CSV/JSON outputs go here
    ├── eval_tokenizer_efficiency.ipynb
    ├── eval_baseline.ipynb
    ├── eval_adapted.ipynb
    ├── requirements.txt
    ├── summary_comparison.md
    └── LAMBDA_SETUP.md
```

## Running the Notebooks

### Option A: Jupyter Lab (Interactive)

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### Option B: Command Line (Batch Execution)

```bash
# Run tokenizer analysis first (no GPU needed)
jupyter nbconvert --to notebook --execute eval_tokenizer_efficiency.ipynb

# Run baseline evaluation
jupyter nbconvert --to notebook --execute eval_baseline.ipynb

# Run adapted model evaluation
jupyter nbconvert --to notebook --execute eval_adapted.ipynb
```

## Notes

- All outputs are confined to `~/projects/tokalign-eval/tokalign-baseline/results/`
- Do NOT modify files outside this directory
- The baseline model is `EleutherAI/pythia-1b` (auto-downloaded from HuggingFace)
- Flash Attention 2 is automatically enabled for H100 (compute capability ≥ 8.0)

