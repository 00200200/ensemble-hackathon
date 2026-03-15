# ECG Digitization and Denoising Pipeline

This repository contains a professional-grade pipeline for digitizing paper ECG records and restoring the signals using State-of-the-Art (SOTA) deep learning methods.

## Overview

The pipeline consists of two main stages:
1.  **Digitizer (`src/ecg_digitizer.py`):** Converts scanned ECG images into 1D digital signals.
    *   **Features:**
        *   Adaptive thresholding and morphological grid removal.
        *   **Viterbi Path Optimization:** Uses a dynamic programming approach (instead of greedy column search) to trace the signal path, ensuring continuity and handling steep QRS complexes robustly.
        *   Automatic lead extraction and layout detection.

2.  **Signal Corrector (`src/signal_corrector.py`):** A neural network to denoise and correct the digitized signals.
    *   **Architecture:** **1D U-Net** (SOTA for signal restoration). 
        *   Replaces standard Residual blocks with a multi-scale Encoder-Decoder architecture with skip connections.
        *   Captures both global context (low frequency) and local details (high frequency).
    *   **Training:** Uses a composite loss function (MSE + L1 + Derivative + Correlation).

## SOTA Improvements

In this version, we have upgraded the core components based on professional ECG processing research (2024-2025 standards):

*   **Viterbi Signal Tracing:** 
    *   *Problem:* Traditional column-wise scanning fails on steep slopes (dashed lines) and noise.
    *   *Solution:* We implemented a vectorized Viterbi algorithm (O(W*H)) that finds the global maximum likelihood path for the signal, minimizing jump penalties while maximizing pixel intensity.

*   **1D U-Net Corrector:**
    *   *Problem:* Simple CNNs struggle with long-range artifacts and preserving high-frequency morphology simultaneously.
    *   *Solution:* A 4-level U-Net architecture allows the model to "see" the entire beat context while maintaining sample-level precision via skip connections.

## Directory Structure
- `data/`: Contains raw images and processed datasets.
- `src/`: Source code.
- `example_submission.py`: Example script to run the full pipeline.

## Usage

### Digitization
To batch convert images to `.npz` files:
```bash
python src/batch_digitize_train.py --images-dir data/train --out-dir data/out --max-images 100
```

### Training the Corrector
```bash
python src/signal_corrector.py train --npz-glob "data/out/*_digitized.npz" --epochs 50 --batch-size 64
```

### Applying the Corrector
```bash
python src/signal_corrector.py apply --model-path data/out/signal_corrector.pt --input-npz input.npz --output-npz output.npz
```
