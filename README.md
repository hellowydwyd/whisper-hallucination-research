**English** | [中文](README_CN.md)

# Whisper ASR Hallucination Research

This project investigates the hallucination problem in OpenAI's Whisper speech recognition model and evaluates various mitigation strategies.

## Introduction

Whisper generates irrelevant text outputs when processing non-speech audio (silence, noise, environmental sounds) - a phenomenon known as "hallucination". This study systematically validates the issue and compares mitigation approaches including VAD preprocessing and BoH post-processing.

## Project Structure

```
├── data/                    # Datasets
│   ├── non_speech/esc50/    # ESC-50 environmental sounds
│   └── speech/librispeech/  # LibriSpeech speech dataset
├── output/                  # Results (CSV, PNG, JSON)
├── report/                  # Paper
│   ├── final_report.tex     # LaTeX source
│   └── final_report.pdf     # Compiled PDF
└── src/                     # Source code
    ├── config.py            # Configuration
    ├── requirements.txt     # Dependencies
    ├── experiments/         # Experiment modules
    ├── models/              # Model wrappers
    ├── scripts/             # Run scripts
    └── utils/               # Utilities
```

## Setup

```bash
# Create conda environment
conda create -n whisper python=3.9
conda activate whisper

# Install dependencies
pip install -r src/requirements.txt
```

## Run Experiments

```bash
# 1. Download datasets
python src/scripts/download_datasets.py

# 2. Synthetic audio hallucination test
python src/scripts/run_experiment.py

# 3. ESC-50 environmental sound test
python src/scripts/run_esc50_experiment.py

# 4. LibriSpeech WER evaluation
python src/scripts/run_librispeech_wer.py

# 5. Mitigation comparison
python src/scripts/run_mitigation_comparison.py
```

## Key Results

| Experiment | Result |
|------------|--------|
| Synthetic Audio Hallucination Rate | 100% |
| ESC-50 Hallucination Rate | 62.5% |
| LibriSpeech WER | 3.71% |
| BoH Mitigation Effect | 74.1% reduction |

## References

1. Radford, A., et al. Robust Speech Recognition via Large-Scale Weak Supervision. ICML, 2022.
2. Wang, Y., et al. Calm-Whisper: Reduce Whisper Hallucination. Interspeech, 2025.
3. Barański, M., et al. Investigation of Whisper ASR Hallucinations. ICASSP, 2025.
