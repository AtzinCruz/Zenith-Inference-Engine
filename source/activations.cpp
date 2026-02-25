#include "../headers/activations.hpp"

void ReLU(Matrix& m) {
    for(double& v : m) {
        if(v < 0.0) {
            v = 0.0;
        }
    }
}
