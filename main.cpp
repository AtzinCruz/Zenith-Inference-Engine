#include <iostream>
#include "headers/DeepLearning.hpp"
#include "headers/matrix.hpp"

using namespace std;

int main() {
    cout << "=== Zenith Deep Learning Engine - XOR Problem ===\n" << endl;

    // Create the basic layers
    DL network;
    network.add_layer(2, 4, true);   // Layer 1: 2 -> 4 neurons with ReLU
    network.add_layer(4, 4, true);   // Layer 2: 4 -> 4 neurons with ReLU
    network.add_layer(4, 1, false);  // output Layer: 4 -> 1 neuron (lineal)

    // Data for XOR Problem
    Matrix x1(2, 1); x1(0, 0) = 0.0; x1(1, 0) = 0.0;  // [0, 0]
    Matrix x2(2, 1); x2(0, 0) = 0.0; x2(1, 0) = 1.0;  // [0, 1]
    Matrix x3(2, 1); x3(0, 0) = 1.0; x3(1, 0) = 0.0;  // [1, 0]
    Matrix x4(2, 1); x4(0, 0) = 1.0; x4(1, 0) = 1.0;  // [1, 1]

    Matrix y0(1, 1); y0(0, 0) = 0.0;  // Output: 0
    Matrix y1(1, 1); y1(0, 0) = 1.0;  // Output: 1

    // 3. train
    cout << "Iniciando entrenamiento...\n" << endl;
    double learning_rate = 0.01;  
    int epochs = 20000; 

    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        Matrix pred1 = network.forward(x1);
        network.backward(network.mse_gradient(pred1, y0), learning_rate);

        Matrix pred2 = network.forward(x2);
        network.backward(network.mse_gradient(pred2, y1), learning_rate);

        Matrix pred3 = network.forward(x3);
        network.backward(network.mse_gradient(pred3, y1), learning_rate);

        Matrix pred4 = network.forward(x4);
        network.backward(network.mse_gradient(pred4, y0), learning_rate);

        if (epoch % 2000 == 0) {
            double loss = (network.mse_loss(pred1, y0) + 
                          network.mse_loss(pred2, y1) + 
                          network.mse_loss(pred3, y1) + 
                          network.mse_loss(pred4, y0)) / 4.0;
            cout << "Epoch " << epoch << " | Loss: " << loss 
                 << " | Preds: [" << pred1(0,0) << ", " << pred2(0,0) 
                 << ", " << pred3(0,0) << ", " << pred4(0,0) << "]" << endl;
        }
    }
    cout << "\n=== Resultados Finales ===" << endl;
    Matrix result1 = network.forward(x1);
    Matrix result2 = network.forward(x2);
    Matrix result3 = network.forward(x3);
    Matrix result4 = network.forward(x4);

    cout << "0 XOR 0 = " << result1(0, 0) << " (esperado: 0.0)" << endl;
    cout << "0 XOR 1 = " << result2(0, 0) << " (esperado: 1.0)" << endl;
    cout << "1 XOR 0 = " << result3(0, 0) << " (esperado: 1.0)" << endl;
    cout << "1 XOR 1 = " << result4(0, 0) << " (esperado: 0.0)" << endl;

    return 0;
}