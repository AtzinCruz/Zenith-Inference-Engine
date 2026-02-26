#include "../headers/activations.hpp"

void ReLU(Matrix& m) {
    for(double& v : m) {
        if(v < 0.0) {
            v = 0.0;
        }
    }
}

void ReLU_derivative(Matrix& m){
    for(double& v : m){
        v = (v > 0.0) ? 1.0 : 0.0;
    }
}

void ReLU_derivative(Matrix& delta, const Matrix& activations) {
    if (delta.get_rows() * delta.get_cols() != activations.get_rows() * activations.get_cols()) {
        throw std::invalid_argument("Dimensiones no coinciden en ReLU_derivative");
    }

    auto it_delta = delta.begin();
    auto it_act = activations.begin();
    
    while(it_delta != delta.end()){        
        *it_delta *= (*it_act > 0.0 ? 1.0 : 0.0);
        ++it_delta;
        ++it_act;
    }
}
