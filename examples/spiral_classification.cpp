/**
 * Spiral Classification - Advanced Deep Neural Network
 * 
 * This example demonstrates a complex multi-layer architecture learning
 * to classify spiral patterns - a highly non-linear problem that requires
 * deep networks to solve effectively.
 * 
 * Challenge: Points arranged in multiple spiral arms need to be classified
 * based on which spiral they belong to.
 */

#include "../headers/DeepLearning.hpp"
#include "../headers/matrix.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>

// Generate spiral dataset
struct DataPoint {
    double x, y;
    double label;  // 0 or 1 for each spiral
};

std::vector<DataPoint> generate_spiral_data(int points_per_spiral, double noise = 0.1) {
    std::vector<DataPoint> data;
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> dist(0.0, noise);
    
    for (int spiral = 0; spiral < 2; ++spiral) {
        for (int i = 0; i < points_per_spiral; ++i) {
            double t = static_cast<double>(i) / points_per_spiral * 4.0 * M_PI;
            double r = t / (4.0 * M_PI) * 5.0;
            
            double angle = t + spiral * M_PI;
            double x = r * std::cos(angle) + dist(gen);
            double y = r * std::sin(angle) + dist(gen);
            
            data.push_back({x, y, static_cast<double>(spiral)});
        }
    }
    
    // Shuffle the data
    std::shuffle(data.begin(), data.end(), gen);
    return data;
}

// Generate complex circular patterns (concentric rings)
std::vector<DataPoint> generate_circular_data(int points_per_class, double noise = 0.05) {
    std::vector<DataPoint> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::normal_distribution<> dist(0.0, noise);
    std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
    
    // Inner circle (class 0)
    for (int i = 0; i < points_per_class; ++i) {
        double angle = angle_dist(gen);
        double r = 1.0 + dist(gen);
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        data.push_back({x, y, 0.0});
    }
    
    // Outer circle (class 1)
    for (int i = 0; i < points_per_class; ++i) {
        double angle = angle_dist(gen);
        double r = 2.5 + dist(gen);
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        data.push_back({x, y, 1.0});
    }
    
    std::shuffle(data.begin(), data.end(), gen);
    return data;
}

// Generate XOR-like quadrant pattern (more complex version)
std::vector<DataPoint> generate_quadrant_data(int points_per_quadrant, double noise = 0.1) {
    std::vector<DataPoint> data;
    std::random_device rd;
    std::mt19937 gen(456);
    std::uniform_real_distribution<> pos_dist(0.2, 2.0);
    std::normal_distribution<> noise_dist(0.0, noise);
    
    // Quadrants with alternating labels (XOR pattern)
    for (int i = 0; i < points_per_quadrant; ++i) {
        // Quadrant 1 (++) -> label 0
        data.push_back({pos_dist(gen) + noise_dist(gen), pos_dist(gen) + noise_dist(gen), 0.0});
        // Quadrant 2 (-+) -> label 1
        data.push_back({-pos_dist(gen) + noise_dist(gen), pos_dist(gen) + noise_dist(gen), 1.0});
        // Quadrant 3 (--) -> label 0
        data.push_back({-pos_dist(gen) + noise_dist(gen), -pos_dist(gen) + noise_dist(gen), 0.0});
        // Quadrant 4 (+-) -> label 1
        data.push_back({pos_dist(gen) + noise_dist(gen), -pos_dist(gen) + noise_dist(gen), 1.0});
    }
    
    std::shuffle(data.begin(), data.end(), gen);
    return data;
}

double calculate_accuracy(DL& network, const std::vector<DataPoint>& data) {
    int correct = 0;
    for (const auto& point : data) {
        Matrix input(2, 1);
        input(0, 0) = point.x;
        input(1, 0) = point.y;
        
        Matrix pred = network.forward(input);
        int predicted_class = (pred(0, 0) > 0.5) ? 1 : 0;
        
        if (predicted_class == static_cast<int>(point.label)) {
            correct++;
        }
    }
    return static_cast<double>(correct) / data.size() * 100.0;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   ZENITH DEEP LEARNING ENGINE - ADVANCED CLASSIFICATION   ║" << std::endl;
    std::cout << "║          Complex Non-Linear Pattern Recognition           ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;

    // Let user choose the problem
    std::cout << "Select classification problem:" << std::endl;
    std::cout << "  1. Spiral Classification (most challenging)" << std::endl;
    std::cout << "  2. Concentric Circles (medium difficulty)" << std::endl;
    std::cout << "  3. Quadrant Patterns (XOR extended)" << std::endl;
    std::cout << "\nEnter choice (1-3, default=1): ";
    
    int choice = 1;
    std::cin.clear();
    if (std::cin.peek() != '\n') {
        std::cin >> choice;
    }
    std::cin.ignore(10000, '\n');
    
    // Generate dataset based on choice
    std::vector<DataPoint> train_data;
    std::string problem_name;
    
    switch(choice) {
        case 2:
            train_data = generate_circular_data(150);
            problem_name = "Concentric Circles";
            break;
        case 3:
            train_data = generate_quadrant_data(50);
            problem_name = "Quadrant Patterns";
            break;
        default:
            train_data = generate_spiral_data(100);
            problem_name = "Spiral Classification";
            break;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Problem: " << problem_name << std::endl;
    std::cout << "Dataset size: " << train_data.size() << " points" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;

    // Create a DEEP neural network architecture
    DL network;
    
    // Input layer: 2 features (x, y coordinates)
    // Multiple hidden layers with varying neurons
    // Output layer: 1 neuron (binary classification)
    
    std::cout << "Building Deep Neural Network Architecture:" << std::endl;
    network.add_layer(2, 16, true);    // Layer 1: 2  -> 16 (ReLU)
    std::cout << "  Layer 1: [2  →  16] (ReLU)" << std::endl;
    
    network.add_layer(16, 32, true);   // Layer 2: 16 -> 32 (ReLU)
    std::cout << "  Layer 2: [16 →  32] (ReLU)" << std::endl;
    
    network.add_layer(32, 32, true);   // Layer 3: 32 -> 32 (ReLU)
    std::cout << "  Layer 3: [32 →  32] (ReLU)" << std::endl;
    
    network.add_layer(32, 16, true);   // Layer 4: 32 -> 16 (ReLU)
    std::cout << "  Layer 4: [32 →  16] (ReLU)" << std::endl;
    
    network.add_layer(16, 8, true);    // Layer 5: 16 -> 8  (ReLU)
    std::cout << "  Layer 5: [16 →   8] (ReLU)" << std::endl;
    
    network.add_layer(8, 1, false);    // Layer 6: 8  -> 1  (Linear)
    std::cout << "  Layer 6: [8  →   1] (Linear/Sigmoid)" << std::endl;
    
    std::cout << "\nTotal Layers: 6 (Deep Architecture)" << std::endl;
    
    // Training configuration
    const double learning_rate = 0.005;
    const int epochs = 15000;
    const int display_interval = 1000;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Training Configuration:" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch Processing: Stochastic Gradient Descent" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;

    // Training loop
    std::cout << "Training Progress:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        // Train on each sample (SGD)
        for (const auto& point : train_data) {
            Matrix input(2, 1);
            input(0, 0) = point.x;
            input(1, 0) = point.y;
            
            Matrix target(1, 1);
            target(0, 0) = point.label;
            
            // Forward pass
            Matrix prediction = network.forward(input);
            
            // Calculate loss
            total_loss += network.mse_loss(prediction, target);
            
            // Backward pass
            Matrix gradient = network.mse_gradient(prediction, target);
            network.backward(gradient, learning_rate);
        }
        
        // Display progress
        if (epoch % display_interval == 0) {
            double avg_loss = total_loss / train_data.size();
            double accuracy = calculate_accuracy(network, train_data);
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Epoch " << std::setw(5) << epoch 
                     << " | Loss: " << std::setw(10) << avg_loss
                     << " | Accuracy: " << std::setprecision(2) << std::setw(6) << accuracy << "%";
            
            // Visual progress bar
            int bar_length = 20;
            int progress = (accuracy / 100.0) * bar_length;
            std::cout << " [";
            for (int i = 0; i < bar_length; ++i) {
                if (i < progress) std::cout << "█";
                else std::cout << "░";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << std::string(60, '-') << std::endl;

    // Final evaluation
    double final_accuracy = calculate_accuracy(network, train_data);
    
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                     TRAINING COMPLETE                      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nFinal Training Accuracy: " << std::fixed << std::setprecision(2) 
              << final_accuracy << "%" << std::endl;

    // Test on some specific points
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Sample Predictions (first 10 training points):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int i = 0; i < std::min(10, static_cast<int>(train_data.size())); ++i) {
        const auto& point = train_data[i];
        Matrix input(2, 1);
        input(0, 0) = point.x;
        input(1, 0) = point.y;
        
        Matrix pred = network.forward(input);
        int predicted_class = (pred(0, 0) > 0.5) ? 1 : 0;
        int actual_class = static_cast<int>(point.label);
        
        std::cout << std::setprecision(3);
        std::cout << "Point (" << std::setw(6) << point.x << ", " << std::setw(6) << point.y << ") "
                  << "→ Pred: " << predicted_class 
                  << " (raw: " << std::setprecision(4) << pred(0, 0) << ") "
                  << "| True: " << actual_class;
        
        if (predicted_class == actual_class) {
            std::cout << " [OK]" << std::endl;
        } else {
            std::cout << " X" << std::endl;
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
    
    // Network statistics
    std::cout << "\nNetwork Statistics:" << std::endl;
    std::cout << "  * Architecture: 2->16->32->32->16->8->1" << std::endl;
    std::cout << "  * Total layers: 6" << std::endl;
    std::cout << "  * Hidden layers: 5" << std::endl;
    std::cout << "  * Training samples: " << train_data.size() << std::endl;
    std::cout << "  * Problem complexity: Non-linear, " << problem_name << std::endl;
    
    if (final_accuracy >= 95.0) {
        std::cout << "\n*** Excellent! The network successfully learned the complex pattern!" << std::endl;
    } else if (final_accuracy >= 85.0) {
        std::cout << "\n** Good! The network learned most of the pattern." << std::endl;
        std::cout << "  Consider training longer or adjusting the learning rate." << std::endl;
    } else {
        std::cout << "\n* The network needs more training." << std::endl;
        std::cout << "  Try increasing epochs or adjusting the architecture." << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n" << std::endl;

    return 0;
}
