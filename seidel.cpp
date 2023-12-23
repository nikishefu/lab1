#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Convergence condition
bool converge(const VectorXd& xk, const VectorXd& xkp, double eps) {
    return (xk - xkp).norm() < eps;
}

// Round to specified number of decimal places
double roundNumber(double x, double eps) {
    int i = 0;
    double neweps = eps;
    while (neweps < 1) {
        i++;
        neweps *= 10;
    }
    int decimalPlaces = pow(10, i);
    x = int(x * decimalPlaces + 0.5) / double(decimalPlaces);
    return x;
}

// Check for diagonal dominance
bool diagonal(const MatrixXd& a, int n) {
    for (int i = 0; i < n; i++) {
        double sum = a.row(i).sum() - abs(a(i, i));
        if (sum > abs(a(i, i))) {
            cout << a(i, i) << " < " << sum << endl;
            return false;
        }
        else {
            cout << a(i, i) << " > " << sum << endl;
        }
    }
    return true;
}

// Function to solve the system using Gauss-Seidel method
VectorXd solveGaussSeidel(const MatrixXd& a, const VectorXd& b, double eps) {
    int n = a.rows();
    VectorXd x = VectorXd::Ones(n);
    VectorXd p(n);

    do {
        p = x;
        for (int i = 0; i < n; i++) {
            double var = (a.row(i) * x).sum() - a(i, i) * x(i);
            x(i) = (b(i) - var) / a(i, i);
        }
    } while (!converge(x, p, eps));

    return x;
}

int main() {

    MatrixXd A(4, 4);
    VectorXd b(4);

    A << 10, -1, 2, 0,
         -1, 11, -1, 3,
         2, -1, 10, -1,
         0, 3, -1, 8;

    b << 6, 25, -11, 15;

    cout << "Diagonal dominance: " << endl;

    if (diagonal(A, A.rows())) {
        VectorXd solution = solveGaussSeidel(A, b, 1.0E-7);

        cout << "Solution:" << endl;
        cout << solution << endl;
        
        VectorXd eigenSolution = A.fullPivLu().solve(b);
        cout << "Solution (Eigen):\n" << eigenSolution << endl;

        cout << "Error:\n" << eigenSolution - solution << endl;
    }
    else {
        cout << "No diagonal dominance" << endl;
    }


    return 0;
}
