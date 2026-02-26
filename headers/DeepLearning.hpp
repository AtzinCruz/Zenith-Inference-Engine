#ifndef DEEPLEARNING_HPP
#define DEEPLEARNING_HPP

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "matrix.hpp"
#include "activations.hpp"

using namespace std;

class Layer{
private:
    Matrix weights;
    Matrix biases;
    Matrix last_inputs;
    Matrix last_z;  
    bool use_activation;

public:
    Layer(int inputs, int neurons, bool activate = true);

    // Forward propagation
    Matrix forward(const Matrix& input);

    // Backward propagation
    Matrix backward(const Matrix& grad_output, double learning_rate);

    // Getters
    Matrix& get_weights() { return weights; }
    Matrix& get_biases() { return biases; }
    const Matrix& get_weights() const { return weights; }
    const Matrix& get_biases() const { return biases; }
    const Matrix& get_last_inputs() const { return last_inputs; }
};

class DL{
private:
    std::vector<Layer> layers;

public:
    // Forward propagation with all the layers
    Matrix forward(const Matrix& input);

    // Backward propagation with all the layers
    void backward(const Matrix& grad_output, double learning_rate);

    // add a layer
    void add_layer(int inputs, int neurons, bool activate = true);

    // Loss functions
    double mse_loss(const Matrix& prediction, const Matrix& target);
    Matrix mse_gradient(const Matrix& prediction, const Matrix& target);

    //Training

    void train(const Matrix& input, const Matrix& target, double lr, int epochs);
};

#endif