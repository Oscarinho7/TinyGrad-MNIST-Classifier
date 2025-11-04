# MNIST Digit Classifier with WebGPU & tinygrad
![Aperçu du projet](images/mnist_img.png)
https://youtu.be/SgQLvMati7A
---

## Overview

This project implements **two neural networks** (MLP and CNN) trained on the **MNIST dataset** using **tinygrad**, then **exported to WebGPU** for **real-time inference directly in the browser**.

You can:
- Draw a digit on a canvas
- Run inference instantly with WebGPU
- See probability bars and the best guess
- Switch between MLP and CNN models

All inference runs **client-side**, **no server**, **no backend**, **no TensorFlow/PyTorch** — just **pure WebGPU + tinygrad**.

---

## Features

- **Interactive canvas** with pen, eraser, brush size
- **Real-time preview** (28×28 pixelated)
- **Auto-run mode** (continuous inference while drawing)
- **Two models**: MLP (96.65%) & CNN (98.36%)
- **WebGPU acceleration** (fast inference ~9ms)
- **Softmax probability bars** for all 10 digits
- **Model switching** without reload
- **HiDPI support** (retina screens)

---

## Model Summary

| Model | Architecture | Accuracy | Loss | Training Time |
|-------|--------------|----------|------|---------------|
| **MLP** | `784 → 512 (SiLU) → 512 (SiLU) → 10` | **96.65%** | 0.18 | ~30s |
| **CNN** | `Conv(/BN → Conv/BN → FC → 10` | **98.36%** | 0.03 | ~102s |

> *Trained with tinygrad, exported to WebGPU via custom `export_model.py`*

[View Full Hyperparameter Log](HYPERPARAMETERS.md)

---

## Setup & Local Run

### Prerequisites
- Python 3.10+
- Node.js (optional, for `http-server`)
- Chrome/Edge with **WebGPU enabled**

## Hyperparameter Log

For full details on all experiments, configurations, and results, see the [HYPERPARAMETERS.md](HYPERPARAMETERS.md) file.




