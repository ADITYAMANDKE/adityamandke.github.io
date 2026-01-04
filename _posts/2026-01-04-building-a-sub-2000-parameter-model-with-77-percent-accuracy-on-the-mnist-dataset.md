---
layout: post
title:  "Building a sub 2000 parameter model with 77% accuracy on the MNIST dataset"
date:   2026-01-04 12:00:00 +0530
categories: machine-learning deep-learning
mathjax: true
---

I recently completed a lab on the Deep-ML website that challenged me to build a tiny classification model with fewer than 2,048 parameters to classify the MNIST dataset. Below, I will explain the problem and walk through my attempts to minimize the weight count.

## The Problem

The goal was to build a model with <2048 parameters that maintains moderately high accuracy on MNIST. The primary task was minimizing the parameter count, as other training variables were fixed:
* **Optimizer**: Adam
* **Learning Rate**: 1e-3
* **Batch Size**: 128
* **Epochs**: 10

---

## A Standard MNIST Classifier

A typical MNIST classifier uses multiple convolution layers followed by a linear layer. Without parameter constraints, the architecture often looks like this:

### PyTorch Implementation
```python
self.features = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pool
    nn.Flatten(),
    nn.Linear(64, 10)
)
```

### Parameter Count Breakdown

* **First Convolution**: 32 x 1 x 3 x 3 = 288 weights + 32 bias parameters = **320 total**.
* **Second Convolution**: 64 x 32 x 3 x 3 = 18,432 weights + 64 bias parameters = **18,496 total**.
* **Fully Connected Layer**: 64 x 10 + 10 bias = **650 total**.
* **Grand Total**: **19,466 parameters**. This model is highly accurate but far exceeds our 2,000-parameter limit.

---

## Attempt #1: Reducing the Parameters

For the first attempt, I reduced the channel counts while keeping the same basic architecture:

| Layer | Details | Formula | Parameters |
|-------|---------|---------|------------|
| Conv1 | Conv(1 \to 8), 3 x 3 | (8 x (1 x 3 x 3) + 8) | 80 |
| Conv2 | Conv(8 \to 16), 3 x 3 | (16 x (8 x 3 x 3) + 16) | 1,168 |
| Linear | Linear(16, 10) | (16 x 10 + 10) | 170 |
| **Total** | — | — | **1,418** |

**Result**: This model gave a relatively low accuracy of ~50%.

---

## Attempt #2: Adding Normalization

Normalization helps the optimizer find the minimum faster, which is critical when limited to 10 epochs. I added Batch Normalization after each convolution layer.

These layers added 2 x 8 + 2 x 16 = 48 extra parameters, bringing the total to **1,466**. This improved the accuracy to 56%, but it was still insufficient.

---

## Attempt #3: Depthwise Separable Convolution

Depthwise separable convolution (depthwise followed by pointwise) drastically reduces computation and parameters while extracting features effectively.

### Final Parameter Count (Bias = False for Convolutions)

| Layer Type | Calculation | Params |
|------------|-------------|--------|
| Standard Conv (Conv1) | 1 x 16 x (3 x 3) | 144 |
| BatchNorm (BN1) | 16 x 2 | 32 |
| Depthwise Conv (Conv2) | 16 x (3 x 3) | 144 |
| BatchNorm (BN2) | 16 x 2 | 32 |
| Pointwise Conv (Conv3) | 16 x 58 x (1 x 1) | 928 |
| BatchNorm (BN3) | 58 x 2 | 116 |
| Linear (FC) | (58 x 10) + 10 | 590 |
| **TOTAL** | — | **1,986** |

---

## Results and Validation

This model reached a final validation accuracy of 76.60%.

| Epoch | Train Loss | Val Loss | Val Score |
|-------|------------|----------|-----------|
| 1 | 2.2200 | 2.3035 | 11.60% |
| 5 | 1.2227 | 1.1704 | 61.20% |
| 10 | 0.7885 | 0.7568 | 76.60% |

---

## Conclusion

This was my first experience building a model this tiny where every single weight matters. Through experimentation with depthwise convolutions and channel counts, I was able to hit the accuracy target within the strict 2,048 parameter budget.
