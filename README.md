
# MCQA Robustness Evaluation

This repository contains tools and results for evaluating robustness in multiple-choice question answering for large audio language models.

The main elements covered now are:

Models:
- Audio Flamingo 2
- Qwen2.5-Omni-7B
- Audio Flamingo 3 (specific transformers version)
- Kimi-Audio-7B-Instruct (requires flash_att)

Benchmarks:
- MMAU
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
- **Original benchmarks**: MMAU-v05.15.25, MMAR, and MMSU datasets
- **Rephrased variants**: Question and answer rephrasing across different models
- **Permuted versions**: Choice order variations for robustness testing

## Models
The evaluation framework supports:
- Audio Flamingo 2 & 3
- Qwen2.5-Omni-7B-Instruct  
- Kimi-Audio-7B-Instruct

## Scripts
- **Generation scripts**: Create rephrased datasets and permutations
- **Evaluation scripts**: Calculate accuracy and consistency metrics
- **Analysis tools**: Compute robustness measures including Consistency Rate (CR) and Consistent and Correct Rate (CCR)

## Evaluation Framework
The repository implements a comprehensive evaluation protocol that:
- Tests choice ordering sensitivity
- Evaluates question/answer rephrasing robustness
- Measures consistency across variations
- Provides detailed robustness metrics beyond simple accuracy
