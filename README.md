# Gelu-Optimization

## Project Overview

The Gaussian Error Linear Unit (GELU) is a widely used activation function in modern deep learning architectures such as **Transformers, CNNs, and EfficientNet**. However, its dependence on the `erf` function makes it computationally expensive, particularly in **CPU-bound or large-scale inference workloads**.

This project explores **activation function optimization strategies** by introducing:

* **CachedGELU** — a lookup-table approximation with interpolation for high-accuracy, low-latency evaluation
* **SnapGELU** — a fast, tunable sigmoid-based alternative for efficient neural network computation

The focus is on balancing:

* **Accuracy vs. Speed**
* **Memory vs. Approximation Error**
* **Practical Deployment vs. Mathematical Fidelity**

---


## Key Observations

*  **~2.23× faster runtime** on a 100 million float CPU benchmark with negligible numerical error
*  Reduced **CIFAR-10 CNN training time by 35.35 seconds over 30 epochs** while maintaining similar accuracy
*  Evaluated **BERT Transformer inference performance** against standard GELU and alternative activations
*  Implements a **lookup-table + interpolation strategy** for constant-time O(1) evaluation
*  Includes **paired t-test validation** for performance reliability

---

##  Repository Structure

```
Gelu-Optimization-Research/
│
├── Algorithm.txt                  # Technical explanation of CachedGELU and grid strategy
├── Cached_gelu.py               # Core implementation of lookup-table based GELU
├── snapgelu.py                 # SnapGELU: parameterized sigmoid-based activation module
│
├── CPU-Benchmarking.ipynb     # Runtime and numerical error benchmarking
├── GELU-CNN-Training.ipynb   # CIFAR-10 CNN training evaluation
├── BERT-Inference-test.ipynb # BERT Transformer inference benchmarking
├── BERT_Graph.ipynb          # Performance and activation visualization
├── Efficientnet_Inference.ipynb # EfficientNet inference comparison
├── GELU-Paired(t-test).ipynb    # Statistical significance testing
│
└── LICENSE
```

---



##  Methodology (CachedGELU)

CachedGELU accelerates GELU evaluation by **precomputing activation values** over a chosen input range (default: **−10 to 10**) into a lookup table with a fixed number of evenly spaced grid points.

### Inference Pipeline

For each input value:

1. **Fractional Index Mapping**
   The input is mapped to a fractional index in the lookup table based on its relative position in the precomputed range.

2. **Linear Interpolation**
   Fast first-order linear interpolation is applied using precomputed neighboring table values and slopes.

3. **Out-of-Range Handling**
   Inputs outside the precomputed range fall back to the **exact GELU formulation** to ensure numerical correctness.

### Properties

*  **Time Complexity:** O(1) per element (constant-time lookup)
*  **Memory–Accuracy Tradeoff:** Adjustable via table size **N**
*  **Vectorized Implementation:** NumPy-based operations for efficient CPU execution

This approach eliminates repeated `erf` calls for in-range inputs, significantly improving throughput while maintaining minimal approximation error.

---


##  Experimental Pipeline

### 1️ CPU Benchmarking

* Measures runtime and numerical error on large-scale synthetic float workloads
* Compares CachedGELU against standard GELU

### 2️ CNN Training (CIFAR-10)

* Trains a CNN using optimized activation functions
* Evaluates epoch-wise training time and final accuracy

### 3️ Transformer Inference (BERT)

* Benchmarks inference latency and throughput on a pretrained BERT model

### 4️ Statistical Validation

* Applies **paired t-tests** to verify consistency and significance of performance gains

---



##  Results Summary

| Experiment          | Outcome                                                  |
| ------------------- | -------------------------------------------------------- |
| CPU Benchmark       | ~2.23× faster runtime with minimal numerical error       |
| CIFAR-10 CNN        | 35.35 seconds reduction over 30 epochs, similar accuracy |
| BERT Inference      | Improved inference performance vs standard GELU          |
| Statistical Testing | Performance gains validated using paired t-test          |

---

##  Applications

* Performance-critical deep learning systems
* CPU-bound inference pipelines
* Model optimization research
* Activation function approximation studies

---

---

⭐ If this repository helped you or inspired your work, consider giving it a star — it helps showcase the impact of this project to recruiters and the research community.
