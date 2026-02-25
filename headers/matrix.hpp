#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <bits/stdc++.h>

using namespace std;

class Matrix{
    private:
    int rows, cols;
    vector<double> data;

    public:
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0){}

    double& operator() (int i, int j);
    double operator() (int i, int j) const;
    
    friend Matrix operator*(const Matrix& A, const Matrix& B);
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
    friend Matrix operator+(const Matrix& A, const Matrix& B);
    
    // Iterators for range-based for loop
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
};

#endif