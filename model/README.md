# TokAlign S2 Checkpoint-2500 Model

This folder contains the TokAlign Stage 2 checkpoint-2500 model weights and tokenizer files.

## Model Details

- **Base Model**: Pythia-1B
- **Tokenizer**: Qwen2-7B tokenizer
- **Training Stage**: Stage 2 (full fine-tuning)
- **Checkpoint**: 2500 steps
- **Parameters**: 1.43B

## Reconstructing the Model

The model weights file has been split into parts to comply with GitHub's 2GB file size limit.

To reconstruct the model, run:

```bash
cd model
cat model.safetensors.part_* > model.safetensors
```

Or on Windows (PowerShell):
```powershell
Get-Content model.safetensors.part_aa, model.safetensors.part_ab -Encoding Byte -ReadCount 0 | Set-Content model.safetensors -Encoding Byte
```

## Usage

After reconstruction, you can load the model with Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
```

## Files

- `model.safetensors.part_aa` - Model weights part 1 (1.5GB)
- `model.safetensors.part_ab` - Model weights part 2 (1.2GB)
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary
- `merges.txt` - BPE merges
- `trainer_state.json` - Training state at checkpoint
- `training_args.bin` - Training arguments

