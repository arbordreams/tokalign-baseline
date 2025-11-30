# TokAlign Baseline Evaluation

This repository contains the TokAlign S2 checkpoint-2500 model and evaluation results for vocabulary adaptation from Pythia-1B to Qwen2 tokenizer.

## Overview

**TokAlign** is an efficient vocabulary adaptation method that replaces the vocabulary of an LLM by learning a one-to-one token alignment matrix from token co-occurrences. This repository contains:

- The trained S2 (Stage 2) checkpoint-2500 model
- Comprehensive evaluation results comparing baseline Pythia-1B vs adapted model
- Visualizations and analysis notebooks
- Paper source files

## Repository Structure

```
├── model/                    # S2 checkpoint-2500 model (Pythia-1B + Qwen2 tokenizer)
├── evaluation/               # All evaluation results
│   ├── results/             # CSV/JSON result files
│   ├── logs/                # Training and evaluation logs
│   ├── run1_partial/        # First evaluation run
│   └── run2_full_results/   # Full evaluation run
├── paper/                    # LaTeX paper source files
├── visualizations/           # Generated figures
├── notebooks/                # Jupyter notebooks for evaluation
├── scripts/                  # Shell and Python scripts
├── RESULTS_SUMMARY.md        # Key findings summary
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/arbordreams/tokalign-baseline.git
cd tokalign-baseline
pip install -r requirements.txt

# Reconstruct the model file (split due to GitHub's 2GB limit)
cd model
cat model.safetensors.part_* > model.safetensors
cd ..
```

### 2. Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Key Results

| Metric | Baseline (Pythia-1B) | Adapted (S2) | Change |
|--------|---------------------|--------------|--------|
| **Fertility** (tokens/word) | 2.059 | 1.874 | **-9.0%** ✅ |
| **Compression Ratio** | 3.169 | 3.501 | **+10.5%** ✅ |
| **Repetition Rate** | 0.443 | 0.285 | **-35.8%** ✅ |
| **chrF++** (MT) | 33.31 | 37.70 | **+13.2%** ✅ |

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed evaluation results.

## Training Details

- **Stage 1**: High learning rate (6.4e-4), embedding-only training
- **Stage 2**: Low learning rate (5e-5), full model fine-tuning
- **Dataset**: Pile corpus, ~20M samples
- **Hardware**: 2x NVIDIA H100 80GB

## Citation

```bibtex
@inproceedings{li-etal-2025-TokAlign,
  author    = {Chong Li and Jiajun Zhang and Chengqing Zong},
  title     = {TokAlign: Efficient Vocabulary Adaptation via Token Alignment},
  booktitle = {Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
}
```

## License

This project is for research purposes.

