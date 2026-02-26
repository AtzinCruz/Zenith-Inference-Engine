/**
 * XOR Problem - Basic Neural Network Example
 * 
 * This example demonstrates how to train a neural network to learn
 * the XOR function, which is not linearly separable.
 * 
 * XOR Truth Table:
 * 0 XOR 0 = 0
 * 0 XOR 1 = 1
 * 1 XOR 0 = 1
 * 1 XOR 1 = 0
 */

#include "../headers/DeepLearning.hpp"
#include "../headers/matrix.hpp"
#include <iostream>

int main() {
    std::cout << "=== Zenith Deep Learning Engine - XOR Problem ===" << std::endl;
    std::cout << "Training a 3-layer network to learn XOR function\n" << std::endl;

    // 1. Create neural network architecture
    DL network;
    network.add_layer(2, 4, true);   // Input: 2 -> Hidden: 4 neurons (ReLU)
    network.add_layer(4, 4, true);   // Hidden: 4 -> Hidden: 4 neurons (ReLU)
    network.add_layer(4, 1, false);  // Hidden: 4 -> Output: 1 neuron (Linear)

    // 2. Prepare XOR training data
    Matrix x1(2, 1); x1(0, 0) = 0.0; x1(1, 0) = 0.0;  // Input: [0, 0]
    Matrix x2(2, 1); x2(0, 0) = 0.0; x2(1, 0) = 1.0;  // Input: [0, 1]
    Matrix x3(2, 1); x3(0, 0) = 1.0; x3(1, 0) = 0.0;  // Input: [1, 0]
    Matrix x4(2, 1); x4(0, 0) = 1.0; x4(1, 0) = 1.0;  // Input: [1, 1]

    Matrix y0(1, 1); y0(0, 0) = 0.0;  // Target: 0
    Matrix y1(1, 1); y1(0, 0) = 1.0;  // Target: 1

    // 3. Training configuration
    const double learning_rate = 0.01;
    const int epochs = 20000;

    std::cout << "Training Parameters:" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Network: [2, 4, 4, 1]\n" << std::endl;

    // 4. Training loop with Stochastic Gradient Descent
    std::cout << "Training progress:" << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train on all examples (one at a time - SGD)
        Matrix pred1 = network.forward(x1);
        network.backward(network.mse_gradient(pred1, y0), learning_rate);

        Matrix pred2 = network.forward(x2);
        network.backward(network.mse_gradient(pred2, y1), learning_rate);

        Matrix pred3 = network.forward(x3);
        network.backward(network.mse_gradient(pred3, y1), learning_rate);

        Matrix pred4 = network.forward(x4);
        network.backward(network.mse_gradient(pred4, y0), learning_rate);

        // Display progress every 2000 epochs
        if (epoch % 2000 == 0) {
            double avg_loss = (network.mse_loss(pred1, y0) + 
                              network.mse_loss(pred2, y1) + 
                              network.mse_loss(pred3, y1) + 
                              network.mse_loss(pred4, y0)) / 4.0;
            
            std::cout << "  Epoch " << epoch << " | Loss: " << avg_loss 
                     << " | Predictions: [" << pred1(0,0) << ", " << pred2(0,0) 
                     << ", " << pred3(0,0) << ", " << pred4(0,0) << "]" << std::endl;
        }
    }

    // 5. Test the trained network
    std::cout << "\n=== Final Results ===" << std::endl;
    
    Matrix result1 = network.forward(x1);
    Matrix result2 = network.forward(x2);
    Matrix result3 = network.forward(x3);
    Matrix result4 = network.forward(x4);

    std::cout << "0 XOR 0 = " << result1(0, 0) << " (expected: 0.0)" << std::endl;
    std::cout << "0 XOR 1 = " << result2(0, 0) << " (expected: 1.0)" << std::endl;
    std::cout << "1 XOR 0 = " << result3(0, 0) << " (expected: 1.0)" << std::endl;
    std::cout << "1 XOR 1 = " << result4(0, 0) << " (expected: 0.0)" << std::endl;

    // Calculate final accuracy
    double final_loss = (network.mse_loss(result1, y0) + 
                        network.mse_loss(result2, y1) + 
                        network.mse_loss(result3, y1) + 
                        network.mse_loss(result4, y0)) / 4.0;
    
    std::cout << "\nFinal Average Loss: " << final_loss << std::endl;

    return 0;
}
