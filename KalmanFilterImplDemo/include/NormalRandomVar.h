#include "KalmanFilterLib/Matrix.h"

#include <Eigen/Eigenvalues>

#include <random>

// Class to generate random multivariate Gaussian noise, copied from:
// https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
template <size_t DIM>
struct NormalRandomVar
{
    Matrix<DIM, DIM> m_transform;

    NormalRandomVar() : NormalRandomVar(Matrix<DIM, DIM>::Zero()) {}
    NormalRandomVar(Matrix<DIM, DIM> cov)
    {
        Eigen::SelfAdjointEigenSolver<Matrix<DIM, DIM>> eigenSolver(cov);
        m_transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Vector<DIM> generate() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<float> dist;

        return m_transform * Vector<DIM>().unaryExpr([&](auto x) { return dist(gen); });
    }

};