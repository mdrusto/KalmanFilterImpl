#pragma once

#include <Eigen/Core>

#include "Matrix.h"
#include "Gaussian.h"

namespace KalmanFilterImpl
{

	void placeholder();

	template <int STATE_DIM, int OUTPUT_DIM, int CONTROL_DIM>
	class KalmanFilter
	{
	public:

		KalmanFilter() = default;
		KalmanFilter(
				const Matrix<STATE_DIM, STATE_DIM>& m_systemMat, 
				const Matrix<STATE_DIM, CONTROL_DIM>& inputMat, 
				const Matrix<OUTPUT_DIM, STATE_DIM>& outputMat, 
				const Matrix<OUTPUT_DIM, CONTROL_DIM>& feedthroughMat, 
				const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, 
				const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) 
				: m_systemMatrix(m_systemMat), m_inputMatrix(inputMat), m_outputMatrix(outputMat), m_feedthroughMatrix(feedthroughMat), 
				m_processNoiseCovariance(processNoiseCov), m_measurementNoiseCovariance(measurementNoiseCov) {}
		~KalmanFilter() = default;

		Gaussian<STATE_DIM> updatePrediction(const Vector<CONTROL_DIM> controlVec, Vector<OUTPUT_DIM>& measurementVec, float deltaTime)
		{
			// Calculate discrete variants of matrices
			Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMatrix, deltaTime);
			Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_inputMatrix, deltaTime);

			// Update a priori estimate
			Vector<STATE_DIM> aPrioriMean = systemMatDiscrete * m_previousEstimate.getMean() + inputMatDiscrete * controlVec;
			Matrix<STATE_DIM, STATE_DIM> aPrioriCov = systemMatDiscrete * m_previousEstimate.getCovariance() * systemMatDiscrete.transpose() + m_processNoiseCovariance;

			// Calculate Kalman gain K
			Matrix<STATE_DIM, OUTPUT_DIM> K =
				aPrioriCov * m_outputMatrix.transpose() * (m_outputMatrix * aPrioriCov * m_outputMatrix.transpose() + m_measurementNoiseCovariance).inverse();

			// Update a posteriori estimate from a priori estimate and current measurement
			Vector<STATE_DIM> aPosterioriMean = aPrioriMean + K * (measurementVec - m_outputMatrix * aPrioriMean);
			//Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov;
			Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov * 
					(Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix).transpose() + K * m_measurementNoiseCovariance * K.transpose();

			Gaussian<STATE_DIM> currentEstimate(aPosterioriMean, aPosterioriCov);

			m_previousEstimate = currentEstimate;

			//std::cout << "Mean: " << aPosterioriMean.transpose() << std::endl;
			//std::cout << "Covariance: " << aPosterioriCov << std::endl;

			return currentEstimate;
		}

	private:

		// Parameters in continuous-time form
		Matrix<STATE_DIM, STATE_DIM> m_systemMatrix;
		Matrix<STATE_DIM, CONTROL_DIM> m_inputMatrix;
		Matrix<OUTPUT_DIM, STATE_DIM> m_outputMatrix;
		Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMatrix;

		Matrix<STATE_DIM, STATE_DIM> m_processNoiseCovariance;
		Matrix<OUTPUT_DIM, CONTROL_DIM> m_measurementNoiseCovariance;

		Gaussian<STATE_DIM> m_previousEstimate{Vector<STATE_DIM>::Zero(), 1000 * Matrix<STATE_DIM, STATE_DIM>::Identity()};

	};

	template <size_t STATE_DIM>
	Matrix<STATE_DIM, STATE_DIM> calculateDiscreteSystemMatrix(const Matrix<STATE_DIM, STATE_DIM>& m_systemMat, float deltaTime)
	{
		return Matrix<STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM) + m_systemMat * deltaTime;
	}

	template <size_t STATE_DIM, size_t CONTROL_DIM>
	Matrix<STATE_DIM, CONTROL_DIM> calculateDiscreteInputMatrix(const Matrix<STATE_DIM, CONTROL_DIM>& inputMat, float deltaTime)
	{
		// Integral approximation method
		//return (calculateDiscreteSystemMatrix(deltaTime) - Matrix<STATE_DIM, STATE_DIM>::Identity()) * m_systemMatrix.inverse() * m_inputMatrix;

		// Euler method
		return inputMat * deltaTime;
	}

}
