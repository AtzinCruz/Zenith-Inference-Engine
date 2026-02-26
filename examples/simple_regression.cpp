/**
 * Simple Linear Regression Example
 * 
 * This example demonstrates training a neural network to learn
 * a simple linear relationship: y = 2x + 1
 * 
 * Network: Single layer (no activation) for linear regression
 */

#include "../headers/DeepLearning.hpp"
#include "../headers/matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Simple Linear Regression: y = 2x + 1 ===" << std::endl;

    // 1. Create a simple linear network (1 input, 1 output, no activation)
    DL network;
    network.add_layer(1, 1, false);  // Linear layer: 1 -> 1 neuron

    // 2. Generate training data: y = 2x + 1
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;
    
    std::cout << "\nGenerating training data:" << std::endl;
    for (int i = 0; i <= 10; ++i) {
        double x = i * 0.5;  // x values: 0, 0.5, 1.0, 1.5, ..., 5.0
        double y = 2.0 * x + 1.0;  // True relationship
        
        Matrix input(1, 1);
        input(0, 0) = x;
        inputs.push_back(input);
        
        Matrix target(1, 1);
        target(0, 0) = y;
        targets.push_back(target);
        
        if (i % 3 == 0) {
            std::cout << "  x = " << x << " -> y = " << y << std::endl;
        }
    }

    // 3. Training configuration
    const double learning_rate = 0.01;
    const int epochs = 1000;

    std::cout << "\nTraining Parameters:" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Training Samples: " << inputs.size() << std::endl;

    // 4. Training loop
    std::cout << "\nTraining progress:" << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        // Train on all examples
        for (size_t i = 0; i < inputs.size(); ++i) {
            Matrix pred = network.forward(inputs[i]);
            Matrix grad = network.mse_gradient(pred, targets[i]);
            network.backward(grad, learning_rate);
            
            total_loss += network.mse_loss(pred, targets[i]);
        }
        
        double avg_loss = total_loss / inputs.size();
        
        if (epoch % 200 == 0) {
            std::cout << "  Epoch " << epoch << " | Avg Loss: " << avg_loss << std::endl;
        }
    }

    // 5. Test the trained network
    std::cout << "\n=== Testing Trained Network ===" << std::endl;
    std::cout << "Testing on training data:" << std::endl;
    
    for (size_t i = 0; i < inputs.size(); i += 2) {
        Matrix pred = network.forward(inputs[i]);
        double x = inputs[i](0, 0);
        double y_true = targets[i](0, 0);
        double y_pred = pred(0, 0);
        
        std::cout << "  x = " << x << " | True: " << y_true 
                 << " | Predicted: " << y_pred 
                 << " | Error: " << (y_true - y_pred) << std::endl;
    }

    // 6. Test on new data (generalization)
    std::cout << "\nTesting on new data:" << std::endl;
    
    std::vector<double> test_values = {2.75, 3.5, 4.25};
    for (double x : test_values) {
        Matrix test_input(1, 1);
        test_input(0, 0) = x;
        
        Matrix pred = network.forward(test_input);
        double y_true = 2.0 * x + 1.0;
        double y_pred = pred(0, 0);
        
        std::cout << "  x = " << x << " | True: " << y_true 
                 << " | Predicted: " << y_pred 
                 << " | Error: " << (y_true - y_pred) << std::endl;
    }

    std::cout << "\n=== Regression Complete ===" << std::endl;
    
    return 0;
}
