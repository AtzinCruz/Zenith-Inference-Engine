#include "../headers/matrix.hpp"

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
    if (A.get_cols() != B.get_rows()) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");
    }

    const int m = A.get_rows();
    const int n = B.get_cols();
    const int k_dim = A.get_cols();

    Matrix result(m, n);  

    const int PARALLEL_THRESHOLD = 128;
    if (m < PARALLEL_THRESHOLD) {
        const double* __restrict a = A.raw_data();
        const double* __restrict b = B.raw_data();
        double* __restrict r = result.raw_data();

        for (int i = 0; i < m; ++i) {
            double* r_row = r + i * n;
            const double* a_row = a + i * k_dim;

            for (int k = 0; k < k_dim; ++k) {
                const double temp = a_row[k];
                const double* b_row = b + k * n;

                for (int j = 0; j < n; ++j) {
                    r_row[j] += temp * b_row[j];
                }
            }
        }

        return result;
    }

    // -------- MULTITHREADED --------

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;

    if (num_threads > static_cast<unsigned int>(m))
        num_threads = m;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    const int rows_per_thread = m / num_threads;

    const double* __restrict a = A.raw_data();
    const double* __restrict b = B.raw_data();
    double* __restrict r = result.raw_data();

    for (unsigned int t = 0; t < num_threads; ++t) {
        const int start_row = t * rows_per_thread;
        const int end_row = (t == num_threads - 1)
                            ? m
                            : (t + 1) * rows_per_thread;

        workers.emplace_back([=]() {
            for (int i = start_row; i < end_row; ++i) {
                double* r_row = r + i * n;
                const double* a_row = a + i * k_dim;

                for (int k = 0; k < k_dim; ++k) {
                    const double temp = a_row[k];
                    const double* b_row = b + k * n;

                    for (int j = 0; j < n; ++j) {
                        r_row[j] += temp * b_row[j];
                    }
                }
            }
        });
    }

    for (auto& thread : workers) {
        thread.join();
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

Matrix& Matrix::operator+=(const Matrix& other) {
    // Broadcasting: si other es un vector columna y this es una matriz
    if (rows == other.rows && other.cols == 1 && cols > 1) {
        for(int i = 0; i < rows * cols; ++i){
            data[i] += other.data[i / cols];
        }
        return *this;
    }
    
    // Suma normal elemento a elemento
    if (cols != other.cols || rows != other.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para sumar");
    }
    for(int i = 0; i < rows * cols; ++i){
        data[i] += other.data[i];
    }
    return *this;
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
    // Broadcasting: si other es un vector columna y this es una matriz
    if (rows == other.rows && other.cols == 1 && cols > 1) {
        for(int i = 0; i < rows * cols; ++i){
            data[i] -= other.data[i / cols];
        }
        return *this;
    }
    
    // Resta normal elemento a elemento
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
        std::vector<double> new_data(rows * cols);
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