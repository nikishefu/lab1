#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
typedef double numType;

// Function to perform Gaussian elimination with partial pivoting
VectorXd gaussElimination(const MatrixXd& A, const VectorXd& b, numType* det = nullptr, int* opCount = nullptr) {
    double innerDet = 1.0;
    int operations = 0;

    // Augmented matrix [A | b]
    MatrixXd augmentedMatrix(A.rows(), A.cols() + 1);
    augmentedMatrix << A, b;

    // Forward elimination with partial pivoting
    for (int i = 0; i < augmentedMatrix.rows() - 1; ++i) {
        // Find the pivot row with the largest absolute value in the current column
        int pivotRow;
        augmentedMatrix.col(i).tail(augmentedMatrix.rows() - i).cwiseAbs().maxCoeff(&pivotRow);
        pivotRow += i;

        // Swap the current row with the pivot row if needed
        if (pivotRow != i) {
            augmentedMatrix.row(i).swap(augmentedMatrix.row(pivotRow));
            innerDet = -innerDet;
        }

        // Update det
        innerDet *= augmentedMatrix(i, i);
        
        // Elimination itself
        for (int j = i + 1; j < augmentedMatrix.rows(); ++j) {
            float factor = augmentedMatrix(j, i) / augmentedMatrix(i, i);
            augmentedMatrix.row(j) -= factor * augmentedMatrix.row(i);

            // Count multiplication and subtraction operations and one division in factor
            operations += (augmentedMatrix.cols() - i) * 2 + 1;
        }
    }
    if (det != nullptr) *det = innerDet * augmentedMatrix(augmentedMatrix.rows() - 1, augmentedMatrix.rows() - 1);

    // Backward substitution
    VectorXd solution(A.cols());
    for (int i = augmentedMatrix.rows() - 1; i >= 0; --i) {
        solution(i) = augmentedMatrix(i, A.cols());
        for (int j = i + 1; j < augmentedMatrix.cols() - 1; ++j) {
            solution(i) -= augmentedMatrix(i, j) * solution(j);

            // Count multiplication and subtraction operations
            operations += 2; // One multiplication and one subtraction per iteration
        }
        solution(i) /= augmentedMatrix(i, i);

        // Count division operations
        operations += 1;
    }

    if (opCount != nullptr) *opCount = operations;
    return solution;
}

void luDecomposition(const MatrixXd& A,
                           MatrixXd& L,
                           MatrixXd& U) {
    int n = A.rows();
    L = MatrixXd::Identity(n, n);
    U = A;

    for (int k = 0; k < n - 1; ++k) {
        for (int i = k + 1; i < n; ++i) {
            L(i, k) = U(i, k) / U(k, k);
            for (int j = k; j < n; ++j) {
                U(i, j) -= L(i, k) * U(k, j);
            }
        }
    }
}

// Perform forward substitution to solve Ly = b
VectorXd forwardSubstitution(const MatrixXd& L, const VectorXd& b) {
    int n = L.rows();
    VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        y(i) = b(i);
        for (int j = 0; j < i; ++j) {
            y(i) -= L(i, j) * y(j);
        }
        y(i) /= L(i, i);
    }

    return y;
}

// Perform back substitution to solve Ux = y
VectorXd backSubstitution(const MatrixXd& U, const VectorXd& y, numType* det=nullptr) {
    int n = U.rows();
    double innerDet = 1;
    VectorXd x(n);

    for (int i = n - 1; i >= 0; --i) {
        x(i) = y(i);
        for (int j = i + 1; j < n; ++j) {
            x(i) -= U(i, j) * x(j);
        }
        x(i) /= U(i, i);
        innerDet *= U(i, i);
    }

    if (det != nullptr) *det = innerDet;

    return x;
}

// Solve the system Ax = b using LU decomposition
VectorXd solveLU(const MatrixXd& A, const VectorXd& b, numType* det=nullptr) {
    MatrixXd L, U;
    luDecomposition(A, L, U);

    // Solve Ly = b
    VectorXd y = forwardSubstitution(L, b);

    // Solve Ux = y
    VectorXd x = backSubstitution(U, y, det);

    return x;
}


void choleskyDecomposition(const MatrixXd& A, MatrixXd& L, int* opCount=nullptr) {
    int n = A.rows();
    int operations = 0;
    L = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) {
                    sum += L(j, k) * L(j, k);
                    operations += 2;
                }
                L(j, j) = sqrt(A(j, j) - sum);
                operations += 2;
            } else {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) {
                    sum += L(i, k) * L(j, k);
                    operations += 2;
                }
                L(i, j) = (A(i, j) - sum) / L(j, j);
                operations += 2;
            }
        }
    }
    if (opCount != nullptr) *opCount = operations;
}

VectorXd solveCholesky(const MatrixXd& A, const VectorXd& b, int* opCount=nullptr) {
    MatrixXd L;
    choleskyDecomposition(A, L, opCount);

    // Solve Ly = b
    VectorXd y = forwardSubstitution(L, b);

    // Solve L^T x = y
    VectorXd x = backSubstitution(L.transpose(), y);

    if (opCount != nullptr) opCount += 2*A.rows()*A.rows();

    return x;
}


numType cond(const MatrixXd& A) {
    return A.norm() * A.inverse().norm();
}


int main() {
    // MatrixXd A(5, 5);
    // A << 97.826801,	 41.998414,	50.609343,	12.433798,	-34.924921,
    //     -73.302177,	-64.404949,	19.807895,	29.211111,	26.857349,
    //     -23.018252,	-99.670959,	90.556130,	-93.071541,	50.149433,
    //      33.468590,	 83.659928,	-78.359246,	66.164907,	99.257844,
    //      91.046960,	 72.786687,	-99.490168,	92.090363,	54.995253;
    // VectorXd b(5);
    // b << 15269.604896,
    //     -9448.144385,
    //     -4763.535748,
    //     178.108181,
    //     6152.954402;

    

    // MatrixXd A(5, 5);
    // A << 322.356557, 65.312142, 31.644208, 39.982007, 98.317556,
    //      65.312142, 92.628208, 5.317508, 19.988782, 125.660547,
    //      31.644208, 5.317508, 205.882681, -20.256120, -84.253834,
    //      39.982007, 19.988782, -20.256120, 69.058051, 57.436709,
    //      98.317556, 125.660547, -84.253834, 57.436709, 274.929621;
    // VectorXd b(5);
    // b << 1096.262200,
    //      2131.120452,
    //      505.422088,
    //      1018.916559,
    //      3445.659083;



    // MatrixXd A(4, 4);
    // A << 4, 1, 0, 0,
    //      2, 5, 3, 0,
    //      0, 3, 6, 2,
    //      0, 0, 1, 7;

    // VectorXd b(4);
    // b << 11, 12, 13, 14;


    
    MatrixXd A(7, 7);
    for (int i = 1; i <= A.rows(); ++i) {
        for (int j = 1; j <= A.cols(); ++j) {
            A(i - 1, j - 1) = 1.0 / (i + j - 1);
        }
    }
    std::cout << "A:" << std::endl;
    std::cout << A << std::endl;

    // x = (1, 1, 1, 1, 1)
    VectorXd b = A.rowwise().sum();
    std::cout << "b:" << std::endl;
    std::cout << b << std::endl;





    numType detGauss, detLU;
    int opCountGauss, opCountCholesky;
    VectorXd solutionGauss = gaussElimination(A, b, &detGauss, &opCountGauss);
    VectorXd solutionLU = solveLU(A, b, &detLU);
    VectorXd solutionCholesky = solveCholesky(A, b, &opCountCholesky);

    // Solve the system using Eigen's built-in solver
    VectorXd solutionEigen = A.colPivHouseholderQr().solve(b);

    // Display the solutions
    std::cout << "\nSolution (Gaussian elimination):\n" << solutionGauss << std::endl;
    std::cout << "\nSolution (Eigen built-in solver):\n" << solutionEigen << std::endl;
    std::cout << "\nSolution (LU):\n" << solutionLU << std::endl;
    std::cout << "\nSolution (Cholesky):\n" << solutionCholesky << std::endl;

    std::cout << "\nNumber of operations (Gauss): " << opCountGauss << std::endl;
    std::cout << "Number of operations (Cholesky): " << opCountCholesky << std::endl;

    numType eigenDet = A.determinant();
    std::cout << "\nDet (Gauss) = " << detGauss << ", Error = " << detGauss - eigenDet << std::endl;
    std::cout << "Det (LU) = " << detLU << ", Error = " << detLU - eigenDet << std::endl;
    std::cout << "Det (Eigen) = " << eigenDet << "\n\n";

    // Compare the solutions
    if (solutionGauss.isApprox(solutionEigen)) {
        std::cout << "Solutions Gauss and by Eigen are approximately equal.\n";
    } else {
        std::cout << "Solutions Gauss and by Eigen are not equal.\n";
    }
    if (solutionLU.isApprox(solutionEigen)) {
        std::cout << "Solutions LU and by Eigen are approximately equal.\n";
    } else {
        std::cout << "Solutions LU and by Eigen are not equal.\n";
    }
    if (solutionCholesky.isApprox(solutionEigen)) {
        std::cout << "Solutions Cholesky and by Eigen are approximately equal.\n";
    } else {
        std::cout << "Solutions Cholesky and by Eigen are not equal.\n";
    }

    std::cout << "Cond: " << cond(A) << std::endl;

    return 0;
}
