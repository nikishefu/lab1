#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/IterativeLinearSolvers>

using namespace Eigen;
using namespace std;

VectorXd solveGaussSeidel(const MatrixXd& A, const VectorXd& b, int maxIterations, double tolerance) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);

    for (int k = 0; k < maxIterations; ++k) {
        VectorXd x_new = x;

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A(i, j) * x_new(j);
                }
            }
            x(i) = (b(i) - sum) / A(i, i);
        }

        // Check for convergence
        if ((x - x_new).norm() < tolerance) {
            cout << "Converged in " << k + 1 << " iterations." << endl;
            break;
        }
    }

    return x;
}

struct IterativeResult {
    VectorXd solution;
    int iterations;
    double priorErrorEstimate;
    double posteriorErrorEstimate;
    double actualError;
};

IterativeResult solveSimpleIteration(const MatrixXd& A, const VectorXd& b, int maxIterations, double tolerance) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);

    IterativeResult result;
    result.iterations = 0;
    result.priorErrorEstimate = 0.0;

    for (int k = 0; k < maxIterations; ++k) {
        VectorXd x_new = x;

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A(i, j) * x_new(j);
                }
            }
            x(i) = (b(i) - sum) / A(i, i);
        }

        // Calculate a priori error estimate (prior to convergence check)
        if (k == 0) {
            result.priorErrorEstimate = (x - x_new).norm();
        }

        // Check for convergence
        if ((x - x_new).norm() < tolerance) {
            cout << "Converged in " << k + 1 << " iterations." << endl;
            result.iterations = k + 1;
            break;
        }
    }

    // Calculate a posteriori error estimate
    result.posteriorErrorEstimate = (A * x - b).norm();

    // Calculate actual error
    result.actualError = (x - A.lu().solve(b)).norm();

    result.solution = x;

    return result;
}

int main() {
    // Example usage
    MatrixXd A(3, 3);
    VectorXd b(3);

    // Set up a system of linear equations
    A << 4, -1, 0,
         -1, 4, -1,
         0, -1, 4;

    b << 15, 10, 10;

    // Solve the system using custom Gauss-Seidel method
    VectorXd zeidelSolution = solveGaussSeidel(A, b, 1000, 1e-6);
    IterativeResult result = solveSimpleIteration(A, b, 1000, 1e-6);

    // Print the results
    cout << "Solution:\n" << result.solution << endl;
    cout << "Number of iterations: " << result.iterations << endl;
    cout << "A priori error estimate: " << result.priorErrorEstimate << endl;
    cout << "A posteriori error estimate: " << result.posteriorErrorEstimate << endl;
    cout << "Actual error: " << result.actualError << endl;

    VectorXd eigenSolution = A.lu().solve(b);

    // Print the solutions
    cout << "Custom Gauss-Seidel Solution:\n" << zeidelSolution << endl;
    cout << "Eigen LU Solution:\n" << eigenSolution << endl;

    return 0;
}

