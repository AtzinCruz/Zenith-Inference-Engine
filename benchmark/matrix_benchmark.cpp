/**
 * Matrix Multiplication Benchmark
 * 
 * Measures performance of the operator* implementation.
 * 
 * Compile: g++ -std=c++17 -O3 -march=native matrix_benchmark.cpp ../source/matrix.cpp -o matrix_bench
 */

#include "../headers/matrix.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std::chrono;

void fill_random(Matrix& m) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < m.get_rows(); ++i) {
        for (int j = 0; j < m.get_cols(); ++j) {
            m(i, j) = dist(gen);
        }
    }
}

double benchmark_multiplication(const Matrix& A, const Matrix& B, int iterations = 10) {
    volatile Matrix warmup = A * B;
    (void)warmup;
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile Matrix result = A * B;
        (void)result;
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    
    return duration / static_cast<double>(iterations);
}

double calculate_gflops(int m, int n, int k, double time_us) {
    double flops = 2.0 * m * n * k;
    double seconds = time_us / 1e6;
    return (flops / seconds) / 1e9;
}

void run_benchmark(int m, int n, int k) {
    std::cout << "\n=== A(" << m << "x" << k << ") * B(" << k << "x" << n 
              << ") = C(" << m << "x" << n << ") ===" << std::endl;
    
    Matrix A(m, k);
    Matrix B(k, n);
    fill_random(A);
    fill_random(B);
    
    std::cout << std::fixed << std::setprecision(2);
    
    double time_us = benchmark_multiplication(A, B, 10);
    double gflops = calculate_gflops(m, n, k, time_us);
    
    std::cout << "   Time:        " << std::setw(10) << time_us << " μs" << std::endl;
    std::cout << "   Performance: " << std::setw(10) << gflops << " GFLOPS" << std::endl;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << " Matrix Multiplication Benchmark (operator*)   " << std::endl;
    std::cout << "================================================" << std::endl;
    
    std::cout << "\nCompiler: ";
    #ifdef __clang__
        std::cout << "Clang " << __clang_major__ << "." << __clang_minor__;
    #elif defined(__GNUC__)
        std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
    #endif
    
    std::cout << " | Optimization: ";
    #ifdef __OPTIMIZE__
        std::cout << "Enabled (-O3)";
    #else
        std::cout << "DISABLED";
    #endif
    std::cout << "\n" << std::endl;
    
    // Small matrices
    run_benchmark(32, 32, 32);
    run_benchmark(64, 64, 64);
    run_benchmark(128, 128, 128);
    run_benchmark(256, 256, 256);
    run_benchmark(512, 512, 512);
    
    // Neural network sizes
    std::cout << "\n=== Casos de redes neuronales ===" << std::endl;
    run_benchmark(1, 784, 128);
    run_benchmark(32, 784, 128);
    run_benchmark(128, 256, 10);
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "             Benchmark Completo                 " << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    return 0;
}