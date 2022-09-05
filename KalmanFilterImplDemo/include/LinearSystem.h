#include "SystemImpl.h"
#include "KalmanFilterLib/LinearKalmanFilter.h"

template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class LinearSystem : public SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>
{
	
public:
    
    virtual Vector<STATE_DIM> calculateNewState(Vector<CONTROL_DIM> controlVec) const override
    {
        
        Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = KalmanFilterImpl::calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMat, this->m_currentDeltaTime);
        Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = KalmanFilterImpl::calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_systemMat, m_inputMat, this->m_currentDeltaTime);

        return systemMatDiscrete * this->m_currentState + inputMatDiscrete * controlVec + this->m_processNoise.generate();
    }

    virtual Vector<OUTPUT_DIM> calculateMeasurement(Vector<CONTROL_DIM> controlVec) const override
    {
        // Return the measurement generated from the current state already calculated from updateAndGetActualState
        return m_outputMat * this->m_currentState + m_feedthroughMat * controlVec + this->m_measurementNoise.generate();
    }

    Matrix<STATE_DIM, STATE_DIM> m_systemMat;
    Matrix<STATE_DIM, CONTROL_DIM> m_inputMat;
    Matrix<OUTPUT_DIM, STATE_DIM> m_outputMat;
    Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMat;

protected:

    virtual void instantiateFilter() override
    {
        this->m_filter = new KalmanFilterImpl::LinearKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(
            m_systemMat, m_inputMat, m_outputMat, m_feedthroughMat, this->m_processNoiseCov, this->m_measurementNoiseCov);
    }
};
