#pragma once

#include <Eigen/Core>

template <int ROWS, int COLS>
using Matrix = Eigen::Matrix<float, ROWS, COLS>;

template <int ROWS>
using Vector = Eigen::Vector<float, ROWS>;