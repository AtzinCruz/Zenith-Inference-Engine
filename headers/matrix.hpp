#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iosfwd>
#include <thread>
#include <stdexcept>
#include <ostream>
#include <utility>

class Matrix{
    private:
    int rows, cols;
    std::vector<double> data;

    public:
    Matrix() : rows(0), cols(0), data() {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0){}

    double& operator() (int i, int j);
    double operator() (int i, int j) const;
    
    //Multiplication operators
    friend Matrix operator*(const Matrix& A, const Matrix& B);
    Matrix& operator*=(const Matrix& other);
    friend Matrix operator*(const Matrix& A, double scalar);
    friend Matrix operator*(double scalar, const Matrix& A);
    Matrix& operator*=(double scalar);
    //Basic operators
    friend Matrix operator+(const Matrix& A, const Matrix& B);
    Matrix& operator+=(const Matrix& other);
    friend Matrix operator-(const Matrix& A, const Matrix& B);
    Matrix& operator-=(const Matrix& other);
    //Ostream operators
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    // Iterators for range-based for loop
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    //Special functions
    Matrix transpose() const;
    Matrix& transpose_();

    // Getters
    inline int get_rows() const { return rows; }
    inline int get_cols() const { return cols; }
    std::size_t size() const { return data.size(); }
    
    // is empty?
    bool empty() const { return data.empty(); }

    double* raw_data() { return data.data(); }
    const double* raw_data() const { return data.data(); }

    // Functions
    Matrix sum_cols();
    
};

#endif