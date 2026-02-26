#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "matrix.hpp"

using namespace std;

void ReLU(Matrix& m);
void ReLU_derivative(Matrix& m);
void ReLU_derivative(Matrix& delta, const Matrix& activations);

#endif