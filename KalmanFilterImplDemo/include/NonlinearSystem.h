#pragma once

#include "SystemImpl.h"

#include "KalmanFilterLib/ExtendedKalmanFilter.h"

template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class NonlinearSystem : public SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>
{

	NonlinearSystem(
		KalmanFilterImpl::ExtendedKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>::SystemUpdateEquationFunc systemUpdateEquationFunc,
		KalmanFilterImpl::ExtendedKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>::ObservationEquationFunc observationEquationFunc,
		const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, 
		const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) : 
		SystemImpl(processNoiseCov, measurementNoiseCov), 
		m_systemUpdateEquationFunc(systemUpdateEquationFunc), m_observationEquationFunc(observationEquationFunc) {}

	void instantiateFilter() override
	{
		this->m_filter = new KalmanFilterImpl::ExtendedKalmanFilter(this->m_processNoiseCov, this->m_measurementNoiseCov);
	}


private:

	KalmanFilterImpl::ExtendedKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>::SystemUpdateEquationFunc m_systemUpdateEquationFunc;
	KalmanFilterImpl::ExtendedKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>::ObservationEquationFunc m_observationEquationFunc;
};
