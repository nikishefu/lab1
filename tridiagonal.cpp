#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

void extractTridiagonalCoefficients(const MatrixXd& A, VectorXd& a, VectorXd& b, VectorXd& c) {
    int n = A.rows();

    a.resize(n - 1);
    b.resize(n);
    c.resize(n - 1);

    for (int i = 0; i < n; ++i) {
        b(i) = A(i, i);
        if (i < n - 1) {
            a(i) = A(i + 1, i);
            c(i) = A(i, i + 1);
        }
    }
}

// Perform tridiagonal matrix algorithm
VectorXd solveExtracted(const VectorXd& lower, VectorXd& main, const VectorXd& upper, VectorXd& b, int* opCount=nullptr) {
    int n = main.size();
    VectorXd x(n);
    int operations = 0;

    // Forward elimination
    for (int i = 1; i < n; ++i) {
        double factor = lower(i - 1) / main(i - 1);
        main(i) -= factor * upper(i - 1);
        b(i) -= factor * b(i - 1);
        operations += 4;
    }

    // Backward substitution
    x(n - 1) = b(n - 1) / main(n - 1);
    for (int i = n - 2; i >= 0; --i) {
        x(i) = (b(i) - upper(i) * x(i + 1)) / main(i);
        operations += 3;
    }

    if (opCount != nullptr) *opCount = operations;

    return x;
}

// Solve the tridiagonal linear system Ax = d
VectorXd solveTridiagonalLinearSystem(const MatrixXd& A, VectorXd& b, int* opCount=nullptr) {
    VectorXd a, diag, c;
    extractTridiagonalCoefficients(A, a, diag, c);
    VectorXd x = solveExtracted(a, diag, c, b, opCount);
    return x;
}

int main() {
    // Example matrix A and vector b
    // MatrixXd A(4, 4);
    // A << 4, 1, 0, 0,
    //      2, 5, 3, 0,
    //      0, 3, 6, 2,
    //      0, 0, 1, 7;

    // VectorXd b(4);
    // b << 11, 12, 13, 14;


    MatrixXd A(5, 5);
    A << 15.051416, 36.724761, 0.000000, 0.000000, 0.000000,
        -39.989346, -15.250037, 60.318093, 0.000000 ,0.000000,
        0.000000, 51.653012, -27.268850, -8.840138 ,0.000000,
        0.000000, 0.000000, 95.980779, -25.703326 ,4.916857,
        0.000000, 0.000000, 0.000000 ,-41.079690, -77.541046;

    VectorXd b(5);
    b << 1575.253205,
        -2471.455608,
        1826.517905,
        -4435.410488,
        3633.564584;





    // Solve using Eigen's built-in solver
    VectorXd xEigen = A.lu().solve(b);

    // Display Eigen's solution
    cout << "Eigen's solution x:\n" << xEigen << "\n";

    int operations;
    VectorXd xCustom = solveTridiagonalLinearSystem(A, b, &operations);

    // Display custom solution
    cout << "Custom solution x:\n" << xCustom << "\n";
    cout << "Number of operations: " << operations << endl;

    // Compare the solutions
    if ((xEigen - xCustom).isZero(1e-9)) {
        cout << "Eigen's solution and custom solution match.\n";
    } else {
        cout << "Eigen's solution and custom solution do not match.\n";
    }

    return 0;
}