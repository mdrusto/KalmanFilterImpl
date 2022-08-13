#pragma once

#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>

#include "Matrix.h"

namespace KalmanFilterImpl
{

	template <int DIM>
	class Gaussian
	{
	public:
		Gaussian() = default;
		Gaussian(Vector<DIM> mean, Matrix<DIM, DIM> covariance)
			: m_meanVec(mean), m_covarianceMat(covariance) {}

		Vector<DIM> getMean() const { return m_meanVec; }
		Matrix<DIM, DIM> getCovariance() const { return m_covarianceMat; }

	private:
		Vector<DIM> m_meanVec;
		Matrix<DIM, DIM> m_covarianceMat;
	};

}
