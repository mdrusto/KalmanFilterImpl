#pragma once

#include "KalmanFilter.h"

#include <functional>

namespace KalmanFilterImpl
{
	template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
	class ExtendedKalmanFilter : public KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>
	{
	public:

		typedef std::function<Matrix<STATE_DIM, STATE_DIM>(Vector<CONTROL_DIM>, double)> SystemUpdateEquationFunc;
		typedef std::function<Matrix<OUTPUT_DIM, STATE_DIM>(Vector<CONTROL_DIM>)> ObservationEquationFunc;

		ExtendedKalmanFilter(
			const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, 
			const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov, 
			SystemUpdateEquationFunc systemUpdateEquationFunc, ObservationEquationFunc observationEquationFunc) :
			KalmanFilter(processNoiseCov, measurementNoiseCov), 
			m_systemUpdateEquationfunc(systemUpdateEquationFunc), m_observationEquationFunc(observationEquationFunc) {}

		Gaussian<STATE_DIM> updatePrediction(const Vector<CONTROL_DIM>& controlVec, const Vector<OUTPUT_DIM>& measurementVec, float deltaTime) override
		{
			// Calculate Jacobians
			Matrix<STATE_DIM> stateTransitionJacobian = calculateStateTransitionJacobian(controlVec);
			Matrix<OUTPUT_DIM, STATE_DIM> observationJacobian = calculateObservationJacobian(controlVec);

			// Calculate the a priori estimate
			Vector<STATE_DIM> aPrioriMean = systemUpdateEquation(controlVec, deltaTime);
			Matrix<STATE_DIM, STATE_DIM> aPrioriCov = stateTransitionJacobian * this->m_previousEstimate.getCovariance() * stateTransitionJacobian.getInverse() + this->m_processNoiseCovariance;

			// Calculate Kalman gain K
			Matrix<STATE_DIM, OUTPUT_DIM> K =
				aPrioriCov * observationJacobian.transpose() * (observationJacobian * aPrioriCov * observationJacobian.transpose() + this->m_measurementNoiseCovariance).inverse();

			// Update a posteriori estimate from a priori estimate and current measurement
			Vector<STATE_DIM> aPosterioriMean = aPrioriMean + K * (measurementVec - observationEquation(controlVec));

			//Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = B * aPrioriCov;
			Matrix<STATE_DIM, STATE_DIM> B = (Matrix<STATE_DIM, STATE_DIM>::Identity() - K * observationJacobian);
			Matrix<STATE_DIM, STATE_DIM> aPosterioriCov = B * aPrioriCov * B.transpose() + K * this->m_measurementNoiseCovariance * K.transpose();

			Gaussian<STATE_DIM> currentEstimate(aPosterioriMean, aPosterioriCov);

			this->m_previousEstimate = currentEstimate;

			//std::cout << "Mean: " << aPosterioriMean.transpose() << std::endl;
			//std::cout << "Covariance: " << aPosterioriCov << std::endl;

			return currentEstimate;
		}

		Vector<STATE_DIM> systemUpdateEquation(const Vector<CONTROL_DIM>& controlVec, float deltaTime) const override
		{
			return systemUpdateEquationFunc(controlVec, deltaTime);
		}

		Vector<OUTPUT_DIM> observationEquation(const Vector<CONTROL_DIM>& controlVec) const override
		{
			return observationEquationFunc(controlVec);
		}


	protected:

		virtual Matrix<STATE_DIM, STATE_DIM> calculateStateTransitionJacobian(const Vector<CONTROL_DIM>& controlVec) const = 0;

		virtual Matrix<OUTPUT_DIM, STATE_DIM> calculateObservationJacobian(const Vector<CONTROL_DIM>& controlVec) const = 0;

	private:

		SystemUpdateEquationFunc m_systemUpdateEquationfunc;
		ObservationEquationFunc m_observationEquationFunc;
	};

}
