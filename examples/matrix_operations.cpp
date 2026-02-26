/**
 * Matrix Operations Demo
 * 
 * This example demonstrates the various matrix operations
 * available in the Zenith engine, including:
 * - Basic arithmetic operations
 * - Broadcasting
 * - Transpose
 * - Matrix multiplication
 */

#include "../headers/matrix.hpp"
#include <iostream>

int main() {
    std::cout << "=== Zenith Matrix Operations Demo ===" << std::endl;

    // 1. Creating matrices
    std::cout << "\n1. Creating Matrices" << std::endl;
    Matrix A(3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    
    std::cout << "Matrix A (3x3):" << std::endl;
    std::cout << A << std::endl;

    Matrix B(3, 1);
    B(0, 0) = 1.0;
    B(1, 0) = 2.0;
    B(2, 0) = 3.0;
    
    std::cout << "Matrix B (3x1):" << std::endl;
    std::cout << B << std::endl;

    // 2. Matrix Addition with Broadcasting
    std::cout << "2. Matrix Addition with Broadcasting" << std::endl;
    std::cout << "A + B (broadcasts B to each column of A):" << std::endl;
    Matrix C = A + B;
    std::cout << C << std::endl;

    // 3. Matrix Multiplication
    std::cout << "3. Matrix Multiplication" << std::endl;
    Matrix D(3, 2);
    D(0, 0) = 1.0; D(0, 1) = 2.0;
    D(1, 0) = 3.0; D(1, 1) = 4.0;
    D(2, 0) = 5.0; D(2, 1) = 6.0;
    
    std::cout << "Matrix D (3x2):" << std::endl;
    std::cout << D << std::endl;
    
    std::cout << "A * D (result is 3x2):" << std::endl;
    Matrix E = A * D;
    std::cout << E << std::endl;

    // 4. Scalar Multiplication
    std::cout << "4. Scalar Multiplication" << std::endl;
    std::cout << "B * 2.0:" << std::endl;
    Matrix F = B * 2.0;
    std::cout << F << std::endl;

    // 5. Transpose
    std::cout << "5. Transpose Operations" << std::endl;
    Matrix G(2, 3);
    G(0, 0) = 1.0; G(0, 1) = 2.0; G(0, 2) = 3.0;
    G(1, 0) = 4.0; G(1, 1) = 5.0; G(1, 2) = 6.0;
    
    std::cout << "Matrix G (2x3):" << std::endl;
    std::cout << G << std::endl;
    
    std::cout << "G.transpose() (3x2):" << std::endl;
    Matrix H = G.transpose();
    std::cout << H << std::endl;

    // 6. In-place Operations
    std::cout << "6. In-place Operations" << std::endl;
    Matrix J(3, 1);
    J(0, 0) = 10.0;
    J(1, 0) = 20.0;
    J(2, 0) = 30.0;
    
    std::cout << "Matrix J before:" << std::endl;
    std::cout << J << std::endl;
    
    J *= 0.5;  // Scale by 0.5
    std::cout << "Matrix J after *= 0.5:" << std::endl;
    std::cout << J << std::endl;
    
    J += B;  // Add B in-place
    std::cout << "Matrix J after += B:" << std::endl;
    std::cout << J << std::endl;

    // 7. Element Access
    std::cout << "7. Element Access" << std::endl;
    std::cout << "Element A(1,1) = " << A(1, 1) << std::endl;
    A(1, 1) = 99.0;
    std::cout << "After setting A(1,1) = 99:" << std::endl;
    std::cout << A << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}
