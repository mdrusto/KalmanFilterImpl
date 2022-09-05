#pragma once

#include <Eigen/Core>

#include "Matrix.h"
#include "Gaussian.h"

namespace KalmanFilterImpl
{

	template <int STATE_DIM, int OUTPUT_DIM, int CONTROL_DIM>
	class KalmanFilter
	{
	public:

		KalmanFilter() = default;
		KalmanFilter(
				const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, 
				const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) :  
				m_processNoiseCovariance(processNoiseCov), m_measurementNoiseCovariance(measurementNoiseCov) {}
		~KalmanFilter() = default;

		virtual Gaussian<STATE_DIM> updatePrediction(const Vector<CONTROL_DIM>& controlVec, const Vector<OUTPUT_DIM>& measurementVec, float deltaTime) = 0;

	protected:
		
		Matrix<STATE_DIM, STATE_DIM> m_processNoiseCovariance;
		Matrix<OUTPUT_DIM, OUTPUT_DIM> m_measurementNoiseCovariance;

		Gaussian<STATE_DIM> m_previousEstimate{Vector<STATE_DIM>::Zero(), 1000 * Matrix<STATE_DIM, STATE_DIM>::Identity()};
	};

	template <size_t STATE_DIM>
	Matrix<STATE_DIM, STATE_DIM> calculateDiscreteSystemMatrix(const Matrix<STATE_DIM, STATE_DIM>& systemMat, float deltaTime)
	{
		return Matrix<STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM) + systemMat * deltaTime;
	}

	template <size_t STATE_DIM, size_t CONTROL_DIM>
	Matrix<STATE_DIM, CONTROL_DIM> calculateDiscreteInputMatrix(const Matrix<STATE_DIM, STATE_DIM>& systemMat, const Matrix<STATE_DIM, CONTROL_DIM>& inputMat, float deltaTime)
	{
		// Integral approximation method
		//return (calculateDiscreteSystemMatrix<STATE_DIM>(systemMat, deltaTime) - Matrix<STATE_DIM, STATE_DIM>::Identity()) * systemMat.inverse() * inputMat;

		// Euler method
		return inputMat * deltaTime;
	}

}
