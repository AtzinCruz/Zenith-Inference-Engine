#include "../headers/matrix.hpp"

double& Matrix::operator()(int i, int j) {
    return data[i * cols + j];
}

double Matrix::operator()(int i, int j) const {
    return data[i * cols + j];
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    // 1. Validar dimensiones (Fundamental en Deep Learning)
    // El número de columnas de A debe ser igual al número de filas de B
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");
    }

    Matrix result(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            double temp = A(i, k); // Guardas el valor en un registro
            for (int j = 0; j < B.cols; ++j) {
                result(i, j) += temp * B(k, j);
            }
        }
    }
    return result;
}

Matrix operator+(const Matrix& A, const Matrix& B){

    if (A.rows == B.rows && B.cols == 1) {
        Matrix result = Matrix(A.rows, A.cols);
        for(int i = 0; i < A.rows * A.cols; ++i){
            result.data[i] = A.data[i] + B.data[i/A.cols];
        }
        return result;
    }


     if (A.cols != B.cols || A.rows != B.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para sumar");
    }
        Matrix result = Matrix(A.rows, B.cols);

    for(int i = 0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] + B.data[i];
    }
    return result;

}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "Matrix(" << m.rows << "x" << m.cols << "):\n";
    
    for (int i = 0; i < m.rows; ++i) {
        os << "[ ";
        for (int j = 0; j < m.cols; ++j) {
            // Usamos el operador () que ya definiste
            os << m(i, j) << " ";
        }
        os << "]\n";
    }
    return os;
}