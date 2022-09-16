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

		Gaussian<STATE_DIM> updatePrediction(const Vector<CONTROL_DIM>& controlVec, const Vector<OUTPUT_DIM>& measurementVec, float deltaTime) override
		{
			// Calculate discrete variants of matrices
			Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMatrix, deltaTime);
			Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_systemMatrix, m_inputMatrix, deltaTime);

			// Calculate the a priori estimate
			Vector<STATE_DIM> aPrioriMean = systemUpdateEquation(controlVec, deltaTime);
			Matrix<STATE_DIM, STATE_DIM> aPrioriCov = systemMatDiscrete * this->m_previousEstimate.getCovariance() * systemMatDiscrete.transpose() + this->m_processNoiseCovariance;

			// Calculate Kalman gain K
			Matrix<STATE_DIM, OUTPUT_DIM> K =
				aPrioriCov * m_outputMatrix.transpose() * (m_outputMatrix * aPrioriCov * m_outputMatrix.transpose() + this->m_measurementNoiseCovariance).inverse();

			// Update a posteriori estimate from a priori estimate and current measurement
			Vector<STATE_DIM> aPosterioriMean = aPrioriMean + K * (measurementVec - observationEquation(controlVec));
			//Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov;
			Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov *
				(Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix).transpose() + K * this->m_measurementNoiseCovariance * K.transpose();

			Gaussian<STATE_DIM> currentEstimate(aPosterioriMean, aPosterioriCov);

			this->m_previousEstimate = currentEstimate;

			//std::cout << "Mean: " << aPosterioriMean.transpose() << std::endl;
			//std::cout << "Covariance: " << aPosterioriCov << std::endl;

			return currentEstimate;
		}

		Vector<STATE_DIM> systemUpdateEquation(const Vector<CONTROL_DIM>& controlVec, float deltaTime) const override
		{
			// Calculate discrete variants of matrices
			Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMatrix, deltaTime);
			Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_systemMatrix, m_inputMatrix, deltaTime);

			// Discrete variant of system update equation
			return systemMatDiscrete * this->m_previousEstimate.getMean() + inputMatDiscrete * controlVec;
		}

		Vector<OUTPUT_DIM> observationEquation(const Vector<CONTROL_DIM>& controlVec) const override
		{
			return this->m_outputMatrix * this->m_previousEstimate.getMean() + this->m_feedthroughMatrix * controlVec;
		}
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
