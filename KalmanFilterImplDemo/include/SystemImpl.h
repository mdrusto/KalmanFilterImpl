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
    ~SystemImpl()
    {
        delete m_filter;
    }

    KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>* m_filter = nullptr;

    const int getStateDim() const { return STATE_DIM; };

    const int getOutputDim() const { return OUTPUT_DIM; };

    const int getControlDim() const { return CONTROL_DIM; };

    // Implement in anonymous subclass
    virtual void setupFilter() = 0;

    void initSystem()
    {
        setupFilter();

        m_processNoise = NormalRandomVar<STATE_DIM>(m_processNoiseCov);
        m_measurementNoise = NormalRandomVar<OUTPUT_DIM>(m_measurementNoiseCov);

        instantiateFilter();
    }

    Vector<STATE_DIM> updateAndGetActualState(Vector<CONTROL_DIM> controlVec)
    {
        const std::chrono::time_point currentFrameTime = std::chrono::high_resolution_clock::now();
        static std::chrono::time_point lastFrameTime = currentFrameTime;
        m_currentDeltaTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentFrameTime - lastFrameTime).count() * 1.0e-9f;
        //m_currentDeltaTime = 0.01f;
        lastFrameTime = currentFrameTime;
        //std::cout << "Current delta time: " << m_currentDeltaTime << " s\n";

        Vector<STATE_DIM> newState = calculateNewState(controlVec);
        m_currentState = newState;
        return newState;
    }

    virtual Vector<OUTPUT_DIM> calculateMeasurement(Vector<CONTROL_DIM> controlVec) const = 0;

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

    virtual Vector<STATE_DIM> calculateNewState(Vector<CONTROL_DIM> controlVec) const = 0;

    virtual void instantiateFilter() = 0;

    NormalRandomVar<STATE_DIM> m_processNoise = NormalRandomVar<STATE_DIM>();
    NormalRandomVar<OUTPUT_DIM> m_measurementNoise = NormalRandomVar<OUTPUT_DIM>();

};
