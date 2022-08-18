#pragma once

#include "KalmanFilterLib/KalmanFilter.h"

#include <Eigen/Eigenvalues>

#include <chrono>
#include <random>
#include <iostream>


template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class SystemImpl {

public:

    SystemImpl() = default;
    ~SystemImpl() = default;

    KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM> m_filter;

    const int getStateDim() const { return STATE_DIM; };

    const int getOutputDim() const { return OUTPUT_DIM; };

    const int getControlDim() const { return CONTROL_DIM; };

    virtual void setupFilter() = 0;

    void createFilter()
    {
        setupFilter();

        m_filter = KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(
            m_systemMat, m_inputMat, m_outputMat, m_feedthroughMat, m_processNoiseCov, m_measurementNoiseCov);

        m_processNoise = NormalRandomVar<STATE_DIM>(m_processNoiseCov);
        m_measurementNoise = NormalRandomVar<OUTPUT_DIM>(m_measurementNoiseCov);
    }

    Vector<STATE_DIM> updateAndGetActualState(Vector<CONTROL_DIM> controlVec)
    {
        const std::chrono::time_point currentFrameTime = std::chrono::high_resolution_clock::now();
        static std::chrono::time_point lastFrameTime = currentFrameTime;
        m_currentDeltaTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentFrameTime - lastFrameTime).count() * 1.0e-9f;
        //m_currentDeltaTime = 0.01f;
        lastFrameTime = currentFrameTime;
        //std::cout << "Current delta time: " << m_currentDeltaTime << " s\n";

        Matrix<STATE_DIM, STATE_DIM> systemMatDiscrete = KalmanFilterImpl::calculateDiscreteSystemMatrix<STATE_DIM>(m_systemMat, m_currentDeltaTime);
        Matrix<STATE_DIM, CONTROL_DIM> inputMatDiscrete = KalmanFilterImpl::calculateDiscreteInputMatrix<STATE_DIM, CONTROL_DIM>(m_systemMat, m_inputMat, m_currentDeltaTime);

        Vector<STATE_DIM> newState = systemMatDiscrete * m_currentState + inputMatDiscrete * controlVec + m_processNoise.generate();
        m_currentState = newState;
        return newState;
    }

    Vector<OUTPUT_DIM> getMeasurement(Vector<CONTROL_DIM> controlVec) const
    {
        // Return the measurement generated from the current state already calculated from updateAndGetActualState
        return m_outputMat * m_currentState + m_feedthroughMat * controlVec + m_measurementNoise.generate();
    }

    KalmanFilterImpl::Gaussian<STATE_DIM> getPrediction(Vector<CONTROL_DIM> controlVec, Vector<OUTPUT_DIM> measurementVec)
    {
        // Use deltaTime calculated already in updateAndGetActualState
        return m_filter.updatePrediction(controlVec, measurementVec, m_currentDeltaTime);
    }

    float m_currentDeltaTime = 0.0f;

    Vector<STATE_DIM> m_currentState;

    Matrix<STATE_DIM, STATE_DIM> m_systemMat;
    Matrix<STATE_DIM, CONTROL_DIM> m_inputMat;
    Matrix<OUTPUT_DIM, STATE_DIM> m_outputMat;
    Matrix<OUTPUT_DIM, CONTROL_DIM> m_feedthroughMat;
    Matrix<STATE_DIM, STATE_DIM> m_processNoiseCov;
    Matrix<OUTPUT_DIM, OUTPUT_DIM> m_measurementNoiseCov;

    // Class to generate random multivariate Gaussian noise, copied from:
    // https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
    template <size_t DIM>
    struct NormalRandomVar
    {
        Matrix<DIM, DIM> m_transform;

        NormalRandomVar() : NormalRandomVar(Matrix<DIM, DIM>::Zero()) {}
        NormalRandomVar(Matrix<DIM, DIM> cov)
        {
            Eigen::SelfAdjointEigenSolver<Matrix<DIM, DIM>> eigenSolver(cov);
            m_transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
        }

        Vector<DIM> generate() const
        {
            static std::mt19937 gen{ std::random_device{}() };
            static std::normal_distribution<float> dist;

            return m_transform * Vector<DIM>().unaryExpr([&](auto x) { return dist(gen); });
        }

    };

    NormalRandomVar<STATE_DIM> m_processNoise = NormalRandomVar<STATE_DIM>();
    NormalRandomVar<OUTPUT_DIM> m_measurementNoise = NormalRandomVar<OUTPUT_DIM>();

};
