#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

// Perform QR decomposition using Gram-Schmidt process
void qr_decomposition(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    int n = A.rows();
    int m = A.cols();

    Q = MatrixXd::Zero(n, m);
    R = MatrixXd::Zero(m, m);

    for (int j = 0; j < m; ++j) {
        VectorXd v = A.col(j);

        for (int i = 0; i < j; ++i) {
            R(i, j) = Q.col(i).dot(A.col(j));
            v -= R(i, j) * Q.col(i);
        }

        R(j, j) = v.norm();
        Q.col(j) = v / R(j, j);
    }
}

// Perform QR algorithm to compute eigenvalues
void qr_algorithm(const MatrixXd& A, MatrixXd& eigenvalues) {
    MatrixXd H = A;
    int n = A.rows();

    for (int i = 0; i < 50; ++i) { // You can adjust the number of iterations
        MatrixXd Q, R;
        qr_decomposition(H, Q, R);
        H = R * Q;
    }

    eigenvalues = H;
}

// Inverse iteration to find an eigenvector corresponding to a given eigenvalue
VectorXd inverse_iteration(const MatrixXd& A, double target_eigenvalue) {
    int n = A.rows();
    MatrixXd I = MatrixXd::Identity(n, n);
    VectorXd b = VectorXd::Random(n); // Initial guess for eigenvector
    double lambda = 0.0;

    for (int i = 0; i < 50; ++i) { // You can adjust the number of iterations
        VectorXd x = (A - target_eigenvalue * I).fullPivLu().solve(b);
        b = x / x.norm();
        lambda = b.dot(A * b) / b.dot(b);
    }

    cout << "Approximated eigenvalue: " << lambda << endl;

    return b;
}

int main() {
    // Example matrix
    MatrixXd A(3, 3);
    A << 4, -2, 1,
         -2, 3, -1,
         1, -1, 2;

    // Compute eigenvalues using QR algorithm
    MatrixXd eigenvalues;
    qr_algorithm(A, eigenvalues);

    cout << "Eigenvalues:\n" << eigenvalues << endl;

    // Find eigenvector using inverse iteration for a specific eigenvalue
    double target_eigenvalue = eigenvalues(0, 0); // Choose any eigenvalue
    VectorXd eigenvector = inverse_iteration(A, target_eigenvalue);

    cout << "Eigenvector:\n" << eigenvector << endl;

    return 0;
}
