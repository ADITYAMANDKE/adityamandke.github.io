---
layout: post
title:  "Building a sub 2000 parameter model with 77% accuracy on the MNIST dataset"
date:   2026-01-04 12:00:00 +0530
categories: machine-learning deep-learning
mathjax: true
---

I recently completed a lab on the Deep-ML website that challenged me to build a tiny classification model with fewer than 2,048 parameters to classify the MNIST dataset. Below, I will explain the problem and walk through my attempts to minimize the weight count.

## The Problem

The goal was to build a model with $<2048$ parameters that maintains moderately high accuracy on MNIST. The primary task was minimizing the parameter count, as other training variables were fixed:
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
Parameter Count BreakdownFirst Convolution: $32 \times 1 \times 3 \times 3 = 288$ weights plus $32$ bias parameters = 320 total.Second Convolution: $64 \times 32 \times 3 \times 3 = 18,432$ weights plus $64$ bias parameters = 18,496 total.Fully Connected Layer: $64 \times 10 + 10$ bias = 650 total.Grand Total: 19,466 parameters. This model is highly accurate but far exceeds our 2,000-parameter limit.Attempt #1: Reducing the ParametersFor the first attempt, I reduced the channel counts while keeping the same basic architecture:LayerDetailsFormulaParametersConv1$Conv(1 \to 8), 3 \times 3$$(8 \times (1 \times 3 \times 3) + 8)$80Conv2$Conv(8 \to 16), 3 \times 3$$(16 \times (8 \times 3 \times 3) + 16)$1,168Linear$Linear(16, 10)$$(16 \times 10 + 10)$170Total1,418Result: This model gave a relatively low accuracy of ~50%.Attempt #2: Adding NormalizationNormalization helps the optimizer find the minimum faster, which is critical when limited to 10 epochs. I added Batch Normalization after each convolution layer.These layers added $2 \times 8 + 2 \times 16 = 48$ extra parameters, bringing the total to 1,466. This improved the accuracy to 56%, but it was still insufficient.Attempt #3: Depthwise Separable ConvolutionDepthwise separable convolution (depthwise followed by pointwise) drastically reduces computation and parameters while extracting features effectively.Final Parameter Count (Bias = False for Convolutions)Layer TypeCalculationParamsStandard Conv (Conv1)$1 \times 16 \times (3 \times 3)$144BatchNorm (BN1)$16 \times 2$32Depthwise Conv (Conv2)$16 \times (3 \times 3)$144BatchNorm (BN2)$16 \times 2$32Pointwise Conv (Conv3)$16 \times 58 \times (1 \times 1)$928BatchNorm (BN3)$58 \times 2$116Linear (FC)$(58 \times 10) + 10$590TOTAL1,986Results and ValidationThis model reached a final validation accuracy of 76.60%.EpochTrain LossVal LossVal Score12.22002.303511.60%51.22271.170461.20%100.78850.756876.60%ConclusionThis was my first experience building a model this tiny where every single weight matters. Through experimentation with depthwise convolutions and channel counts, I was able to hit the accuracy target within the strict 2,048 parameter budget.<style>table {margin-left: auto;margin-right: auto;margin-bottom: 24px;border-collapse: collapse;width: 100%;}th, td {padding: 12px;border: 1px solid #ccc;text-align: left;}th {background-color: #f4f4f4;}tr:nth-child(even) {background-color: #fafafa;}</style>{% if page.mathjax %}<script type="text/javascript" async src="https://www.google.com/search?q=https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js%3Fconfig%3DTeX-MML-AM_CHTML"></script>{% endif %}