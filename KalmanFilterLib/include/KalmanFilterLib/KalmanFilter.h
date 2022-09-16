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


		virtual Vector<STATE_DIM> systemUpdateEquation(const Vector<CONTROL_DIM>& controlVec, float deltaTime) const = 0;
		virtual Vector<OUTPUT_DIM> observationEquation(const Vector<CONTROL_DIM>& controlVec) const = 0;

	protected:
		
		Matrix<STATE_DIM, STATE_DIM> m_processNoiseCovariance;
		Matrix<OUTPUT_DIM, OUTPUT_DIM> m_measurementNoiseCovariance;

		Gaussian<STATE_DIM> m_previousEstimate{Vector<STATE_DIM>::Zero(), 1000 * Matrix<STATE_DIM, STATE_DIM>::Identity()};
	};

}
