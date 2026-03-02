
# MCQA Robustness Evaluation

This repository contains tools and results for evaluating robustness in multiple-choice question answering for large audio language models.

The main elements covered now are:

Models:
- Audio Flamingo 2
- Qwen2.5-Omni-7B
- Audio Flamingo 3 (specific transformers version)
- Kimi-Audio-7B-Instruct (requires flash_att)

Benchmarks:
- MMAU-v05.15.25
- MMAR
- MMSU

## Environment setup
To be able to replicate results, execute next lines:

```bash
python3 -m venv .venv
source .venv/bin/activate/
pip install -r requirements.txt
```

# Repository Contents

This repository contains:

## Data
We put here the different versions of the benchmarks (json): original, rephrased questions, rephrased distractors and mix or perturbations.

## Models
The evaluation framework supports:
- Audio Flamingo 2 & 3
- Qwen2.5-Omni-7B-Instruct  
- Kimi-Audio-7B-Instruct

## Scripts
- **Run permutations**: Create rephrased datasets, ordering perturbations or mix of perturbations
- **Inference**: Inference code per each model.
- **Evaluation**: Compute robustness measures: Consistency Rate (CR) and Consistent and Correct Rate (CCR)
