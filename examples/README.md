# Zenith Examples

This directory contains example programs demonstrating various features and use cases of the Zenith Inference Engine.

## 📁 Examples Overview

### 1. `xor_basic.cpp` - XOR Problem
**Difficulty:** Beginner  
**Concepts:** Basic neural network, training loop, forward/backward pass

Learn how to train a neural network to solve the classic XOR problem, which requires non-linear decision boundaries.

**Compile & Run:**
```bash
g++ -std=c++17 xor_basic.cpp ../source/*.cpp -o xor_demo
./xor_demo
```

**Expected Output:**
```
0 XOR 0 ≈ 0.01
0 XOR 1 ≈ 0.99
1 XOR 0 ≈ 0.99
1 XOR 1 ≈ 0.01
```

---

### 2. `matrix_operations.cpp` - Matrix Library Demo
**Difficulty:** Beginner  
**Concepts:** Matrix creation, arithmetic, broadcasting, transpose

Explore all available matrix operations including addition, multiplication, broadcasting, and in-place operations.

**Compile & Run:**
```bash
g++ -std=c++17 matrix_operations.cpp ../source/matrix.cpp -o matrix_demo
./matrix_demo
```

**Key Features Demonstrated:**
- Matrix creation and initialization
- Element access
- Broadcasting (adding vector to matrix)
- Matrix multiplication
- Transpose operations
- In-place modifications

---

### 3. `simple_regression.cpp` - Linear Regression
**Difficulty:** Beginner  
**Concepts:** Linear regression, single-layer network, generalization

Train a network to learn a simple linear relationship: y = 2x + 1

**Compile & Run:**
```bash
g++ -std=c++17 simple_regression.cpp ../source/*.cpp -o regression_demo
./regression_demo
```

**What You'll Learn:**
- Training data generation
- Single-layer networks (linear)
- Testing on unseen data
- Model generalization

---

### 4. `custom_architecture.cpp` - Network Architectures
**Difficulty:** Intermediate  
**Concepts:** Network design, architecture patterns, layer configuration

Explore different network architectures and learn when to use each one.

**Compile & Run:**
```bash
g++ -std=c++17 custom_architecture.cpp ../source/*.cpp -o architecture_demo
./architecture_demo
```

**Architectures Covered:**
- Shallow networks (1 hidden layer)
- Deep networks (multiple hidden layers)
- Wide networks (many neurons per layer)
- Bottleneck/autoencoder style
- Multi-output networks

---

## 🛠️ Building All Examples

To compile all examples at once:

```bash
# From the examples/ directory
for file in *.cpp; do
    [ "$file" = "README.md" ] && continue
    output="${file%.cpp}_demo"
    echo "Compiling $file..."
    g++ -std=c++17 "$file" ../source/*.cpp -o "$output"
done
```

Or create a simple build script:

```bash
#!/bin/bash
# build_examples.sh

g++ -std=c++17 xor_basic.cpp ../source/*.cpp -o xor_demo
g++ -std=c++17 matrix_operations.cpp ../source/matrix.cpp -o matrix_demo
g++ -std=c++17 simple_regression.cpp ../source/*.cpp -o regression_demo
g++ -std=c++17 custom_architecture.cpp ../source/*.cpp -o architecture_demo

echo "All examples compiled successfully!"
```

## 📚 Learning Path

**Recommended order for beginners:**

1. **matrix_operations.cpp** - Understand the matrix library
2. **simple_regression.cpp** - Learn basic training concepts
3. **xor_basic.cpp** - Work with non-linear problems
4. **custom_architecture.cpp** - Explore network design

## 🎯 Next Steps

After completing these examples, try:

- Modifying hyperparameters (learning rate, epochs, layers)
- Creating your own dataset
- Experimenting with different network architectures
- Implementing new features (batch training, momentum, etc.)

## 📝 Notes

- All examples are standalone and can be compiled independently
- Examples use relative paths to access headers and source files
- C++17 or later required for compilation
- No external dependencies needed

## 🐛 Troubleshooting

**Compilation errors:**
- Ensure you're in the `examples/` directory
- Check that all source files exist in `../source/`
- Verify C++17 support: `g++ --version`

**Runtime issues:**
- Check that learning rate isn't too high (causes divergence)
- Verify network architecture matches data dimensions
- Ensure sufficient training epochs for convergence

## 💡 Tips

- Start with small networks and gradually increase complexity
- Monitor loss during training - it should decrease
- If loss plateaus, try adjusting learning rate or architecture
- Use the debugging output to understand what's happening

---

Happy learning with Zenith! 🚀
