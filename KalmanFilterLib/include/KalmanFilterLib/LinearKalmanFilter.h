#pragma once

#include "KalmanFilter.h"

namespace KalmanFilterImpl
{
	
	template <int STATE_DIM, int OUTPUT_DIM, int CONTROL_DIM>
	class LinearKalmanFilter : public KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>
	{

	public:

		// Parameters in continuous-time form
		Matrix<STATE_DIM, STATE_DIM> m_systemMatrix;
		Matrix<STATE_DIM, CONTROL_DIM> m_inputMatrix;
		Matrix<OUTPUT_DIM, STATE_DIM> m_outputMatrix;
		Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMatrix;

		LinearKalmanFilter(
			const Matrix<STATE_DIM, STATE_DIM>& m_systemMat,
			const Matrix<STATE_DIM, CONTROL_DIM>& inputMat,
			const Matrix<OUTPUT_DIM, STATE_DIM>& outputMat,
			const Matrix<OUTPUT_DIM, CONTROL_DIM>& feedthroughMat,
			const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov,
			const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) : 
			KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(processNoiseCov, measurementNoiseCov), 
			m_systemMatrix(m_systemMat), m_inputMatrix(inputMat), m_outputMatrix(outputMat), m_feedthroughMatrix(feedthroughMat) {}

		virtual Gaussian<STATE_DIM> updatePrediction(const Vector<CONTROL_DIM>& controlVec, const Vector<OUTPUT_DIM>& measurementVec, float deltaTime) override
		{
			// Calculate discrete variants of matrices
			Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMatrix, deltaTime);
			Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_systemMatrix, m_inputMatrix, deltaTime);

			// Calculate the a priori estimate
			Vector<STATE_DIM> aPrioriMean = systemMatDiscrete * this->m_previousEstimate.getMean() + inputMatDiscrete * controlVec;
			Matrix<STATE_DIM, STATE_DIM> aPrioriCov = systemMatDiscrete * this->m_previousEstimate.getCovariance() * systemMatDiscrete.transpose() + this->m_processNoiseCovariance;

			// Calculate Kalman gain K
			Matrix<STATE_DIM, OUTPUT_DIM> K =
				aPrioriCov * m_outputMatrix.transpose() * (m_outputMatrix * aPrioriCov * m_outputMatrix.transpose() + this->m_measurementNoiseCovariance).inverse();

			// Update a posteriori estimate from a priori estimate and current measurement
			Vector<STATE_DIM> aPosterioriMean = aPrioriMean + K * (measurementVec - m_outputMatrix * aPrioriMean);
			//Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov;
			Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov *
				(Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix).transpose() + K * this->m_measurementNoiseCovariance * K.transpose();

			Gaussian<STATE_DIM> currentEstimate(aPosterioriMean, aPosterioriCov);

			this->m_previousEstimate = currentEstimate;

			//std::cout << "Mean: " << aPosterioriMean.transpose() << std::endl;
			//std::cout << "Covariance: " << aPosterioriCov << std::endl;

			return currentEstimate;
		}

	};
}
