#pragma once

#include "KalmanFilterLib/KalmanFilter.h"

#include <Eigen/Eigenvalues>


template <size_t STATE_DIM, size_t OUTPUT_DIM, size_t CONTROL_DIM>
class SystemImpl {

public:

    SystemImpl() = default;
    ~SystemImpl()
    {
        delete filter;
        delete processNoise;
        delete measurementNoise;
    }

    KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>* filter = nullptr;

    const int getStateDim() const { return STATE_DIM };

    const int getOutputDim() const { return OUTPUT_DIM };

    const int getControlDim() const { return CONTROL_DIM };

    virtual void setupFilter() = 0;

    void createFilter()
    {
        setupFilter();

        filter = new KalmanFilterImpl::KalmanFilter<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>(
            systemMat, inputMat, outputMat, feedthroughMat, processNoiseCov, measurementNoiseCov);

        processNoise = new NormalRandomVar<STATE_DIM>(processNoiseCov);
        measurementNoise = new NormalRandomVar<OUTPUT_DIM>(measurementNoiseCov);
    }

    Vector<STATE_DIM> updateAndGetActualState(Vector<CONTROL_DIM> controlVec)
    {
        Vector<STATE_DIM> newState = systemMat * currentState + inputMat * controlVec + processNoise->generate();
        currentState = newState;
        return newState;
    }

    Vector<OUTPUT_DIM> getMeasurement(Vector<CONTROL_DIM> controlVec)
    {
        return outputMat * currentState + feedthroughMat * controlVec + measurementNoise->generate();
    }

    KalmanFilterImpl::Gaussian<STATE_DIM> getPrediction(Vector<CONTROL_DIM> controlVec, Vector<OUTPUT_DIM> measurementVec)
    {
        return filter->updatePrediction(controlVec, measurementVec);
    }

    Vector<CONTROL_DIM> initialControlVec;

protected:

    Matrix<STATE_DIM, STATE_DIM> systemMat;
    Matrix<STATE_DIM, CONTROL_DIM> inputMat;
    Matrix<OUTPUT_DIM, STATE_DIM> outputMat;
    Matrix<OUTPUT_DIM, CONTROL_DIM> feedthroughMat;
    Matrix<STATE_DIM, STATE_DIM> processNoiseCov;
    Matrix<OUTPUT_DIM, OUTPUT_DIM> measurementNoiseCov;

    Vector<STATE_DIM> currentState;

    // Class to generate random multivariate Gaussian noise, copied from:
    // https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
    template <size_t DIM>
    struct NormalRandomVar
    {
        Matrix<DIM, DIM> m_transform;

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

    NormalRandomVar<STATE_DIM> *processNoise;
    NormalRandomVar<OUTPUT_DIM> *measurementNoise;

};
