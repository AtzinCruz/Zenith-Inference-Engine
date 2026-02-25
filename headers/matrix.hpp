#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iosfwd>

using namespace std;

class Matrix{
    private:
    int rows, cols;
    vector<double> data;

    public:
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
    Matrix transpose_() const;
};

#endif