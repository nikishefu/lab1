#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

struct PowerIterationResult {
    double eigenvalue;
    VectorXd eigenvector;
    int iterations;
};

PowerIterationResult powerIteration(const MatrixXd& A, double tolerance, int maxIterations) {
    int n = A.rows();
    VectorXd x = VectorXd::Random(n);  // Initial random guess

    PowerIterationResult result;
    result.iterations = 0;

    for (int k = 0; k < maxIterations; ++k) {
        VectorXd x_new = A * x;

        // Normalize the vector
        x_new.normalize();

        // Calculate the eigenvalue as the Rayleigh quotient
        double lambda = x_new.transpose() * A * x_new;

        // Check for convergence
        if ((x_new - x).norm() < tolerance) {
            cout << "Converged in " << k + 1 << " iterations." << endl;
            result.iterations = k + 1;
            result.eigenvalue = lambda;
            result.eigenvector = x_new;
            break;
        }

        x = x_new;
    }

    return result;
}

int main() {
    // Example usage
    MatrixXd A(3, 3);

    // Set up a symmetric matrix (for simplicity)
    A << 4, -1, 2,
         -1, 5, -1,
         2, -1, 6;

    // Set the desired tolerance and maximum number of iterations
    double tolerance = 1e-6;
    int maxIterations = 1000;

    // Find the eigenvalue with the largest magnitude and its corresponding eigenvector
    PowerIterationResult result = powerIteration(A, tolerance, maxIterations);

    // Print the results
    cout << "Eigenvalue: " << result.eigenvalue << endl;
    cout << "Eigenvector:\n" << result.eigenvector << endl;
    cout << "Number of iterations: " << result.iterations << endl;

    return 0;
}
