/**
 * Custom Architecture Example
 * 
 * This example demonstrates building neural networks with
 * different architectures and configurations.
 */

#include "../headers/DeepLearning.hpp"
#include "../headers/matrix.hpp"
#include <iostream>

void demo_shallow_network() {
    std::cout << "\n=== Shallow Network (1 Hidden Layer) ===" << std::endl;
    
    DL network;
    network.add_layer(2, 3, true);   // 2 -> 3 (ReLU)
    network.add_layer(3, 1, false);  // 3 -> 1 (Linear)
    
    std::cout << "Architecture: [2, 3, 1]" << std::endl;
    std::cout << "Layers: 2" << std::endl;
    std::cout << "Total parameters: 2*3 + 3 + 3*1 + 1 = 13" << std::endl;
}

void demo_deep_network() {
    std::cout << "\n=== Deep Network (3 Hidden Layers) ===" << std::endl;
    
    DL network;
    network.add_layer(4, 8, true);    // 4 -> 8 (ReLU)
    network.add_layer(8, 6, true);    // 8 -> 6 (ReLU)
    network.add_layer(6, 4, true);    // 6 -> 4 (ReLU)
    network.add_layer(4, 1, false);   // 4 -> 1 (Linear)
    
    std::cout << "Architecture: [4, 8, 6, 4, 1]" << std::endl;
    std::cout << "Layers: 4" << std::endl;
    std::cout << "Depth: Good for complex patterns" << std::endl;
}

void demo_wide_network() {
    std::cout << "\n=== Wide Network (Many Neurons Per Layer) ===" << std::endl;
    
    DL network;
    network.add_layer(3, 16, true);   // 3 -> 16 (ReLU)
    network.add_layer(16, 16, true);  // 16 -> 16 (ReLU)
    network.add_layer(16, 1, false);  // 16 -> 1 (Linear)
    
    std::cout << "Architecture: [3, 16, 16, 1]" << std::endl;
    std::cout << "Layers: 3" << std::endl;
    std::cout << "Width: More capacity per layer" << std::endl;
}

void demo_bottleneck_network() {
    std::cout << "\n=== Bottleneck Network (Encoder-Decoder Style) ===" << std::endl;
    
    DL network;
    network.add_layer(8, 6, true);    // 8 -> 6 (ReLU) - Compress
    network.add_layer(6, 3, true);    // 6 -> 3 (ReLU) - Bottleneck
    network.add_layer(3, 6, true);    // 3 -> 6 (ReLU) - Expand
    network.add_layer(6, 8, false);   // 6 -> 8 (Linear) - Output
    
    std::cout << "Architecture: [8, 6, 3, 6, 8]" << std::endl;
    std::cout << "Layers: 4" << std::endl;
    std::cout << "Use case: Dimensionality reduction, autoencoders" << std::endl;
}

void demo_binary_classification() {
    std::cout << "\n=== Binary Classification Network ===" << std::endl;
    
    DL network;
    network.add_layer(5, 8, true);    // 5 features -> 8 (ReLU)
    network.add_layer(8, 4, true);    // 8 -> 4 (ReLU)
    network.add_layer(4, 1, false);   // 4 -> 1 (Linear, can use sigmoid later)
    
    std::cout << "Architecture: [5, 8, 4, 1]" << std::endl;
    std::cout << "Layers: 3" << std::endl;
    std::cout << "Output: Single value for binary decision" << std::endl;
    std::cout << "Note: Add sigmoid activation for 0-1 output" << std::endl;
}

void demo_multi_output() {
    std::cout << "\n=== Multi-Output Network ===" << std::endl;
    
    DL network;
    network.add_layer(3, 6, true);    // 3 inputs -> 6 (ReLU)
    network.add_layer(6, 6, true);    // 6 -> 6 (ReLU)
    network.add_layer(6, 4, false);   // 6 -> 4 outputs (Linear)
    
    std::cout << "Architecture: [3, 6, 6, 4]" << std::endl;
    std::cout << "Layers: 3" << std::endl;
    std::cout << "Output: 4 values (e.g., multi-class or multi-task)" << std::endl;
}

int main() {
    std::cout << "=== Zenith Neural Network Architectures Demo ===" << std::endl;
    std::cout << "\nExploring different network configurations:\n" << std::endl;

    demo_shallow_network();
    demo_deep_network();
    demo_wide_network();
    demo_bottleneck_network();
    demo_binary_classification();
    demo_multi_output();

    std::cout << "\n=== Architecture Guidelines ===" << std::endl;
    std::cout << "\n1. Depth vs Width:" << std::endl;
    std::cout << "   - Deep: Better for hierarchical features" << std::endl;
    std::cout << "   - Wide: Better for capacity at each level" << std::endl;
    
    std::cout << "\n2. Layer Sizes:" << std::endl;
    std::cout << "   - Usually decrease: [10, 8, 6, 4, 2]" << std::endl;
    std::cout << "   - Or constant: [8, 8, 8]" << std::endl;
    std::cout << "   - Avoid large jumps in size" << std::endl;
    
    std::cout << "\n3. Activation Functions:" << std::endl;
    std::cout << "   - ReLU (true): Hidden layers" << std::endl;
    std::cout << "   - Linear (false): Output layer for regression" << std::endl;
    std::cout << "   - Future: Sigmoid for 0-1 output" << std::endl;

    std::cout << "\n4. Number of Parameters:" << std::endl;
    std::cout << "   - More parameters = more capacity" << std::endl;
    std::cout << "   - But risk of overfitting on small datasets" << std::endl;
    std::cout << "   - Start small, increase if needed" << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}
