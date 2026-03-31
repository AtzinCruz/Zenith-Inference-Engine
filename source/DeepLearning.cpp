#include "../headers/DeepLearning.hpp"

Layer::Layer(int inputs, int neurons, bool activate) 
    : weights(neurons, inputs), biases(neurons, 1), use_activation(activate) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double std_dev = activate ? std::sqrt(2.0 / inputs) : std::sqrt(1.0 / inputs);
    std::normal_distribution<double> dist(0.0, std_dev);
    
    for(auto &w : weights){
        w = dist(gen);
    }
    for(auto &b : biases){
        b = activate ? 0.1 : 0.0;
    }
}

Matrix Layer::forward(const Matrix& input) {
    last_inputs = input;
    last_z = weights * input;
    
    int N = input.get_cols();
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < biases.get_rows(); ++i){
            last_z(i, j) += biases(i, 0);
        }
    }

    Matrix output = last_z;
    if(use_activation){
        ReLU(output);
    }
    return output;
}

Matrix Layer::backward(const Matrix& grad_output, double learning_rate) {
    int N = last_inputs.get_cols();
    Matrix delta = grad_output;

    if(use_activation){
        ReLU_derivative(delta, last_z);
    }

    Matrix grad_input = weights.transpose() * delta;
    Matrix dw = (delta * last_inputs.transpose() * (1.0 / N));
    Matrix db = delta.sum_cols() * (1.0 / N);

    weights -= (dw * learning_rate);
    biases -= (db * learning_rate);

    return grad_input;
}

void DL::add_layer(int inputs, int neurons, bool activate){
    layers.emplace_back(inputs, neurons, activate);
}

Matrix DL::forward(const Matrix& input){
    Matrix current_input = input;

    for(auto &layer : layers){
        current_input = layer.forward(current_input);
    }
    return current_input;
}

void DL::backward(const Matrix& grad_output, double learning_rate){
    Matrix current_grad = grad_output;

    for (int i = layers.size() - 1; i >= 0; --i) {
        current_grad = layers[i].backward(current_grad, learning_rate);
    }
}

double DL::mse_loss(const Matrix& prediction, const Matrix& target) {
    double error = 0.0;
    auto it_p = prediction.begin();
    auto it_t = target.begin();
    int n = 0;

    while (it_p != prediction.end()) {
        double diff = *it_p - *it_t;
        error += diff * diff;
        ++it_p; ++it_t; ++n;
    }

    return error / n;
}
Matrix DL::mse_gradient(const Matrix& prediction, const Matrix& target) {
    Matrix grad = prediction - target;
    return grad;
}

void DL::train(const Matrix& input, const Matrix& target, double lr, int epochs){
    for(int i = 0; i < epochs; ++i){
        Matrix prediction = this->forward(input);
        if(i % 100 == 0){
            double loss = this->mse_loss(prediction, target);
            cout << "Epoch " << i << " - Error (MSE): " << loss << endl;
        }

        Matrix grad = mse_gradient(prediction, target);

        this->backward(grad, lr);
    }
}
