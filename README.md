# Zenith-Inference-Engine

A Deep Learning inference engine built from scratch in **C++20**, focused on core matrix operations and a modular structure that can scale over time.

## Current status

The project currently includes:

- `Matrix` class with linear memory storage (`std::vector<double>`).
- Indexed access through `operator()(i, j)`.
- Operators:
	- Matrix multiplication (`operator*`).
	- Matrix addition (`operator+`).
	- Column-vector bias broadcasting (`A + b`, where `b` is `n x 1`).
- Readable console printing (`operator<<`).
- Element-wise `ReLU(Matrix&)` activation.
- A working example in `DeepLearning.cpp` that creates matrices and runs multiplication.

## Project structure

```text
Zenith-Inference-Engine/
├── DeepLearning.cpp
├── headers/
│   ├── matrix.hpp
│   └── activations.hpp
└── source/
	├── matrix.cpp
	└── activations.cpp
```

## Requisitos

## Requirements

- A compiler with C++20 support (`g++` recommended).
- Linux/macOS/WSL (also portable to Windows with toolchain adjustments).

## Build

From the project root:

```bash
g++ -std=c++20 -O2 DeepLearning.cpp source/matrix.cpp source/activations.cpp -o zenith_engine
```

## Run

```bash
./zenith_engine
```

## Design choices

- **Linear memory layout**: matrices are stored in a 1D array to improve cache locality.
- **Loop-order optimization**: multiplication uses the `i-k-j` pattern to reuse `A(i, k)`.
- **Module split**:
	- `headers/` contains interfaces.
	- `source/` contains implementations.

## Suggested next steps

- Add more activations (`sigmoid`, `tanh`, `softmax`).
- Add additional operations (`transpose`, `dot`, `hadamard`).
- Add bounds checking for matrix access.
- Integrate unit tests.
- Set up a `CMake`-based build system.

## Note

The current goal of this repository is to establish the engine foundation; it is still in an early development stage.
