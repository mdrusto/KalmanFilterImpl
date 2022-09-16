#include "SystemImpl.h"
#include "KalmanFilterLib/LinearKalmanFilter.h"

template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class LinearSystem : public SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>
{
	
public:
    
    LinearSystem() = default;
    LinearSystem(const Matrix<STATE_DIM, STATE_DIM>& systemMat, const Matrix<STATE_DIM, CONTROL_DIM>& inputMat,
        const Matrix<OUTPUT_DIM, STATE_DIM>& outputMat, const Matrix<OUTPUT_DIM, CONTROL_DIM>& feedthroughMat,
        const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) :
        SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(processNoiseCov, measurementNoiseCov), 
        m_systemMat(systemMat), m_inputMat(inputMat), m_outputMat(outputMat), m_feedthroughMat(feedthroughMat) {}

protected:

    void instantiateFilter() override
    {
        this->m_filter = new KalmanFilterImpl::LinearKalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(
            m_systemMat, m_inputMat, m_outputMat, m_feedthroughMat, this->m_processNoiseCov, this->m_measurementNoiseCov);
    }

private:

    Matrix<STATE_DIM, STATE_DIM> m_systemMat;
    Matrix<STATE_DIM, CONTROL_DIM> m_inputMat;
    Matrix<OUTPUT_DIM, STATE_DIM> m_outputMat;
    Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMat;

};
