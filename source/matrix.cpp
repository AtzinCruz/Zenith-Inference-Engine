#include "../headers/matrix.hpp"
#include <stdexcept>
#include <ostream>
#include <utility>

double& Matrix::operator()(int i, int j) {
    return data[i * cols + j];
}

double Matrix::operator()(int i, int j) const {
    return data[i * cols + j];
}

/*

    Multiplication operators

*/

Matrix operator*(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");
    }

    Matrix result(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            double temp = A(i, k);
            for (int j = 0; j < B.cols; ++j) {
                result(i, j) += temp * B(k, j);
            }
        }
    }
    return result;
}

Matrix& Matrix::operator*=(const Matrix& other){
    if (cols != other.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");
    }
    
    // Crear matriz temporal con el resultado
    Matrix temp = (*this) * other;
    
    // Actualizar dimensiones y datos
    cols = temp.cols;
    rows = temp.rows;
    data = std::move(temp.data);
    
    return *this;
}

Matrix operator*(const Matrix& A, double scalar){
    Matrix result(A.rows, A.cols);
    for(int i = 0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] * scalar;
    }
    return result;
}
Matrix operator*(double scalar, const Matrix& A){
    return A * scalar;
}

Matrix& Matrix::operator*=(double scalar){
    for(int i = 0; i < rows * cols; ++i){
        this->data[i] *= scalar;
    }
    return *this;
}

/*

    Basic Operators (+, -)

*/

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


Matrix operator-(const Matrix& A, const Matrix& B){
     if (A.cols != B.cols || A.rows != B.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para restar");
    }
        Matrix result = Matrix(A.rows, B.cols);

    for(int i = 0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] - B.data[i];
    }
    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (cols != other.cols || rows != other.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para restar");
    }
    for(int i = 0; i < rows * cols; ++i){
        data[i] -= other.data[i];
    }
    return *this;
}


/*

    Ostream operators

*/

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "Matrix(" << m.rows << "x" << m.cols << "):\n";
    
    for (int i = 0; i < m.rows; ++i) {
        os << "[ ";
        for (int j = 0; j < m.cols; ++j) {
            os << m(i, j) << " ";
        }
        os << "]\n";
    }
    return os;
}

/*

    Special functions

*/

Matrix Matrix::transpose() const{
    Matrix result(cols, rows);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix& Matrix::transpose_(){
    if (rows == cols) {
        for(int i = 0; i < rows; ++i){
            for(int j = i + 1; j < cols; ++j){
                std::swap(data[i * cols + j], data[j * cols + i]);
            }
        }
    } else {        
        vector<double> new_data(rows * cols);
        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                new_data[j * rows + i] = data[i * cols + j];
            }
        }
        data = std::move(new_data);
        std::swap(rows, cols);
    }
    return *this;
}