
---

# 🌌 Zenith Inference Engine

**Zenith** is a lightweight, high-performance deep learning inference and training engine built from scratch in **C++17**. Designed under the *First Principles* philosophy, this engine implements its own tensor core, contiguous memory management, and optimization algorithms without relying on third-party libraries like Eigen, OpenBLAS, or PyTorch.

## ✨ Key Highlights

* **🧠 Custom Tensor Core**: Highly optimized matrix library utilizing row-major contiguous memory for cache efficiency.
* **⚡ Zero-Copy Philosophy**: Leverages C++ move semantics (`std::move`) and in-place operations to minimize CPU overhead.
* **🛠️ Deep Learning from Scratch**: Full implementation of vectorized **Backpropagation** and **He Initialization**.
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
└── examples/           # Use cases and demos (XOR, Regression)

```

---

## 🔧 Installation & Quick Start

### Prerequisites

* A **C++17** compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+).

### One-Step Build & Run

```bash
# Compile and run the XOR gate demonstration
g++ -std=c++17 main.cpp source/*.cpp -O3 -o zenith && ./zenith

```

> **Performance Note:** It is highly recommended to use the `-O3` flag. This enables aggressive compiler loop optimizations that significantly boost matrix multiplication throughput.

---

## 🧪 Battle-Tested Example: The XOR Problem

Zenith successfully solves the classic non-linear XOR problem, demonstrating the integrity of the backpropagation chain through hidden layers.

```cpp
#include "headers/DeepLearning.hpp"

int main() {
    DL engine;
    
    // Architecture: 2 (Inputs) -> 4 (Hidden ReLU) -> 1 (Output Linear)
    engine.add_layer(2, 4, true);  
    engine.add_layer(4, 1, false); 

    // Training (Stochastic Gradient Descent)
    for (int i = 0; i < 10000; ++i) {
        engine.train(input_xor, target_xor, 0.05, 1);
    }
    
    // Inference
    Matrix result = engine.forward(input_test);
    std::cout << "Zenith Prediction: " << result(0,0) << std::endl;
}

```

---

## 📈 Technical Deep Dive

### Memory & Cache Optimization

To maximize data throughput, Zenith stores matrices in a **1D contiguous vector**. This significantly reduces cache misses during large-scale matrix multiplications compared to nested structures.

### Gradient Stability

* **He Initialization**: Specifically tuned for ReLU layers to prevent vanishing gradients during early epochs.
* **Pre-activation Caching**: Zenith caches `z` values (pre-activation) to ensure mathematically precise derivative calculations during the backward pass.

---

## 🚀 Future Roadmap

* [ ] **Optimizers Pro**: Integration of Adam and RMSProp.
* [ ] **SIMD Acceleration**: Utilizing AVX/SSE instructions for hardware-level parallelism.
* [ ] **Persistence**: Model serialization to binary format for rapid deployment.
* [ ] **Auto-Diff Engine**: Implementing a computational graph for automatic differentiation.

---

## 👤 Author

**Atzin Eduardo Cruz Briones** *AI Engineering Student @ Universidad Panamericana* *Incoming Systems Software Engineer Intern @ **Oracle** (GoldenGate Core Team)*
