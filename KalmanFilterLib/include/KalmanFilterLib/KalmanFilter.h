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

		KalmanFilter(
				const Matrix<STATE_DIM, STATE_DIM>& systemMat, 
				const Matrix<STATE_DIM, CONTROL_DIM>& inputMat, 
				const Matrix<OUTPUT_DIM, STATE_DIM>& outputMat, 
				const Matrix<OUTPUT_DIM, CONTROL_DIM>& feedthroughMat, 
				const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, 
				const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) 
				: m_systemMatrix(systemMat), m_inputMatrix(inputMat), m_outputMatrix(outputMat), m_feedthroughMatrix(feedthroughMat), 
				m_processNoiseCovariance(processNoiseCov), m_measurementNoiseCovariance(measurementNoiseCov)
		{
			m_previousEstimate = Gaussian<STATE_DIM>(Vector<STATE_DIM>::Zero(), Matrix<STATE_DIM, STATE_DIM>::Zero());
		}
		~KalmanFilter() = default;

		Gaussian<STATE_DIM> updatePrediction(Vector<CONTROL_DIM> controlVec, Vector<OUTPUT_DIM> measurementVec)
		{

			// Update a priori estimate
			Vector<STATE_DIM> aPrioriMean = m_systemMatrix * m_previousEstimate.getMean() + m_inputMatrix * controlVec;
			Matrix<STATE_DIM, STATE_DIM> aPrioriCov = m_systemMatrix * m_previousEstimate.getCovariance() * m_systemMatrix.transpose() + m_processNoiseCovariance;

			Matrix<STATE_DIM, OUTPUT_DIM> K =
				aPrioriCov * m_outputMatrix.transpose() * (m_outputMatrix * aPrioriCov * m_outputMatrix.transpose() + m_measurementNoiseCovariance).inverse();

			// Update a posteriori estimate from a priori estimate and current measurement
			Vector<STATE_DIM> aPosterioriMean = aPrioriMean + K * (measurementVec - m_outputMatrix * aPrioriMean);
			Matrix<STATE_DIM, STATE_DIM> aPosterioriCov =
				(Matrix<STATE_DIM, STATE_DIM>::Identity() - K * m_outputMatrix) * aPrioriCov;

			Gaussian<STATE_DIM> currentEstimate(aPosterioriMean, aPosterioriCov);

			m_previousEstimate = currentEstimate;

			return currentEstimate;
		}

	private:

		Matrix<STATE_DIM, STATE_DIM> m_systemMatrix;
		Matrix<STATE_DIM, CONTROL_DIM> m_inputMatrix;
		Matrix<OUTPUT_DIM, STATE_DIM> m_outputMatrix;
		Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMatrix;

		Matrix<STATE_DIM, STATE_DIM> m_processNoiseCovariance;
		Matrix<OUTPUT_DIM, CONTROL_DIM> m_measurementNoiseCovariance;

		Gaussian<STATE_DIM> m_previousEstimate;

	};

}
