#pragma once

#include "NormalRandomVar.h"

#include "KalmanFilterLib/KalmanFilter.h"
#include "KalmanFilterLib/LinearKalmanFilter.h"

#include <chrono>
#include <iostream>


template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class SystemImpl {

public:

    SystemImpl() = default;
    SystemImpl(const Matrix<STATE_DIM, STATE_DIM>& processNoiseCov, const Matrix<OUTPUT_DIM, OUTPUT_DIM>& measurementNoiseCov) :
        m_processNoiseCov(processNoiseCov), m_measurementNoiseCov(measurementNoiseCov) {}
    ~SystemImpl()
    {
        delete m_filter;
    }

    KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>* m_filter = nullptr;

    const int getStateDim() const { return STATE_DIM; };

    const int getOutputDim() const { return OUTPUT_DIM; };

    const int getControlDim() const { return CONTROL_DIM; };

    virtual void instantiateFilter() = 0;

    void initSystem()
    {
        m_processNoise = NormalRandomVar<STATE_DIM>(m_processNoiseCov);
        m_measurementNoise = NormalRandomVar<OUTPUT_DIM>(m_measurementNoiseCov);

        instantiateFilter();
    }

    Vector<STATE_DIM> updateAndGetActualState(Vector<CONTROL_DIM> controlVec)
    {
        std::cout << controlVec.transpose() << "\n";
        using namespace std::chrono;
        const time_point currentFrameTime = high_resolution_clock::now();
        static time_point lastFrameTime = currentFrameTime;
        m_currentDeltaTime = duration_cast<nanoseconds>(currentFrameTime - lastFrameTime).count() * 1.0e-9f;
        //m_currentDeltaTime = 0.01f;
        lastFrameTime = currentFrameTime;
        //std::cout << "Current delta time: " << m_currentDeltaTime << " s\n";

        Vector<STATE_DIM> newState = calculateNewState(controlVec);
        m_currentState = newState;
        return newState;
    }

    Vector<STATE_DIM> calculateNewState(const Vector<CONTROL_DIM>& controlVec) const
    {
        return m_filter->systemUpdateEquation(controlVec, m_currentDeltaTime) + this->m_processNoise.generate();
    }

    Vector<OUTPUT_DIM> calculateMeasurement(const Vector<CONTROL_DIM>& controlVec) const
    {
        // Return the measurement generated from the current state already calculated from updateAndGetActualState
        return m_filter->observationEquation(controlVec) + this->m_measurementNoise.generate();
    }

    KalmanFilterImpl::Gaussian<STATE_DIM> getPrediction(Vector<CONTROL_DIM> controlVec, Vector<OUTPUT_DIM> measurementVec)
    {
        // Use deltaTime calculated already in updateAndGetActualState
        return m_filter->updatePrediction(controlVec, measurementVec, m_currentDeltaTime);
    }

    float m_currentDeltaTime = 0.0f;

    Vector<STATE_DIM> m_currentState;

    Matrix<STATE_DIM, STATE_DIM> m_processNoiseCov;
    Matrix<OUTPUT_DIM, OUTPUT_DIM> m_measurementNoiseCov;

protected:

    NormalRandomVar<STATE_DIM> m_processNoise = NormalRandomVar<STATE_DIM>();
    NormalRandomVar<OUTPUT_DIM> m_measurementNoise = NormalRandomVar<OUTPUT_DIM>();

};
