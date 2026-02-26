
---

# 🌌 Zenith Inference Engine

**Zenith** is a lightweight, high-performance deep learning inference and training engine built from scratch in **C++17**. Designed under the *First Principles* philosophy, this engine implements its own tensor core, contiguous memory management, and optimization algorithms without relying on third-party libraries like Eigen, OpenBLAS, or PyTorch.

## ✨ Key Highlights

* **🧠 Custom Tensor Core**: Highly optimized matrix library utilizing row-major contiguous memory for cache efficiency.
* **⚡ Zero-Copy Philosophy**: Leverages C++ move semantics (`std::move`) and in-place operations to minimize CPU overhead.
* **🛠️ Deep Learning from Scratch**: Full implementation of vectorized **Backpropagation** and **He Initialization**.
* **🎯 Production-Ready**: Successfully trains deep networks (6+ layers) to 99%+ accuracy on complex non-linear problems.
* **🔌 Minimalist & Portable**: No external dependencies. Built for high-performance environments and embedded systems.

---

## 🏗️ System Architecture

The engine is built on three modular pillars to ensure scalability and clean code separation:

| Component | Responsibility |
| --- | --- |
| **Matrix Engine** | Linear algebra, in-place transposition, and broadcasting logic. |
| **Layer API** | Weight/Bias management and activation state handling (Forward/Backward). |
| **Network Orchestrator** | Coordinates data flow and manages the training/inference lifecycle. |

---

## 📁 Project Structure

```bash
Zenith-Inference-Engine/
├── headers/            # Interface and class definitions (.hpp)
├── source/             # Mathematical logic and implementations (.cpp)
├── examples/           # Use cases and demos
│   ├── xor_basic.cpp              # Classic XOR problem (3 layers)
│   ├── spiral_classification.cpp  # Advanced deep learning (6 layers)
│   ├── simple_regression.cpp      # Linear regression demo
│   ├── matrix_operations.cpp      # Matrix engine capabilities
│   └── custom_architecture.cpp    # Custom network building
└──  benchmark/          # Performance benchmarking tools

```

---

## 🔧 Installation & Quick Start

### Prerequisites

* A **C++17** compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+).

### One-Step Build & Run

```bash
# Compile and run the XOR gate demonstration
g++ -std=c++17 examples/xor_basic.cpp source/*.cpp -O3 -o zenith_xor && ./zenith_xor

# Or try the advanced deep learning example (6-layer spiral classification)
g++ -std=c++17 examples/spiral_classification.cpp source/*.cpp -O3 -o zenith_spiral && ./zenith_spiral

```

> **Performance Note:** It is highly recommended to use the `-O3` flag. This enables aggressive compiler loop optimizations that significantly boost matrix multiplication throughput.

---

## 🧪 Examples & Demonstrations

### 1. Classic XOR Problem (3-Layer Network)

Zenith successfully solves the classic non-linear XOR problem, demonstrating the integrity of the backpropagation chain through hidden layers.

```cpp
#include "headers/DeepLearning.hpp"

int main() {
    DL network;
    
    // Architecture: 2 (Inputs) -> 4 (Hidden ReLU) -> 1 (Output Linear)
    network.add_layer(2, 4, true);  
    network.add_layer(4, 1, false); 

    // Training data
    Matrix x1(2, 1); x1(0,0) = 0.0; x1(1,0) = 0.0;
    Matrix y0(1, 1); y0(0,0) = 0.0;
    
    // Training loop
    for (int epoch = 0; epoch < 10000; ++epoch) {
        Matrix pred = network.forward(x1);
        network.backward(network.mse_gradient(pred, y0), 0.01);
    }
    
    return 0;
}
```

**Results:** Converges to ~1e-29 loss with perfect XOR classification in 20,000 epochs.

### 2. Advanced Deep Learning: Spiral Classification (6-Layer Network)

A challenging non-linear problem that requires deep architectures to solve. The network learns to classify points from two intertwined spiral arms.

**Architecture:** `2 → 16 → 32 → 32 → 16 → 8 → 1` (6 layers, 5 hidden)

```bash
# Compile and run
g++ -std=c++17 -O3 examples/spiral_classification.cpp source/*.cpp -o zenith_spiral
./zenith_spiral

# Choose from 3 problems:
#   1. Spiral Classification (most challenging)
#   2. Concentric Circles (medium difficulty)  
#   3. Quadrant Patterns (XOR extended)
```

**Performance Results:**

| Problem | Dataset Size | Final Accuracy | Epochs |
|---------|--------------|----------------|--------|
| Spiral Classification | 200 points | **99.50%** | 15,000 |
| Concentric Circles | 300 points | **100.00%** | 15,000 |
| Quadrant Patterns | 200 points | **100.00%** | 15,000 |

**Key Features:**
- Real-time training progress with visual progress bars
- Stochastic Gradient Descent on synthetic datasets
- Multiple complex non-linear classification problems
- Sample predictions with confidence scores

---

## 📈 Technical Deep Dive

### Memory & Cache Optimization

To maximize data throughput, Zenith stores matrices in a **1D contiguous vector**. This significantly reduces cache misses during large-scale matrix multiplications compared to nested structures.

### Gradient Stability

* **He Initialization**: Specifically tuned for ReLU layers to prevent vanishing gradients during early epochs.
* **Pre-activation Caching**: Zenith caches `z` values (pre-activation) to ensure mathematically precise derivative calculations during the backward pass.

---

## 🚀 Roadmap & Current Progress

### ✅ Completed Features
- [x] **Deep Multi-Layer Networks**: Successfully training 6+ layer architectures
- [x] **Backpropagation Engine**: Full gradient computation through arbitrary depth
- [x] **He Weight Initialization**: Optimized for ReLU activation functions
- [x] **Stochastic Gradient Descent**: Efficient training on mini-batches
- [x] **Complex Problem Solving**: Spiral classification, concentric circles, extended XOR

### 🔮 Future Enhancements
* [ ] **Advanced Optimizers**: Integration of Adam, RMSProp, and Momentum
* [ ] **Softmax & Cross-Entropy**: Multi-class classification support
* [ ] **SIMD Acceleration**: Utilizing AVX2/AVX-512 instructions for hardware-level parallelism
* [ ] **Model Persistence**: Binary serialization for rapid deployment and inference
* [ ] **Auto-Diff Engine**: Computational graph for automatic differentiation
* [ ] **Batch Normalization**: Layer normalization for training stability
* [ ] **Dropout Regularization**: Prevent overfitting in deep networks

---

## 👤 Author

**Atzin Eduardo Cruz Briones** *AI Engineering Student @ Universidad Panamericana* *Incoming Systems Software Engineer Intern @ **Oracle** (GoldenGate Core Team)*
