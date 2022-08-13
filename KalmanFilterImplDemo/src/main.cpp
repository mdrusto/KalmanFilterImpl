#include <Eigen/Core>

#include "KalmanFilterLib/KalmanFilter.h"
#include "KalmanFilterLib/Gaussian.h"

#include "SystemImpl.h"

#include <stdio.h>
#include <random>
#include <chrono>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <implot.h>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

const ImVec4 CLEAR_COLOUR = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

struct ScrollingBuffer {

    int m_maxSize;
    int m_offset;
    ImVector<ImVec2> Data;

    ScrollingBuffer(int max_size = 10000) {
        m_maxSize = max_size;
        m_offset = 0;
        Data.reserve(m_maxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < m_maxSize)
            Data.push_back(ImVec2(x, y));
        else {
            Data[m_offset] = ImVec2(x, y);
            m_offset = (m_offset + 1) % m_maxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            m_offset = 0;
        }
    }
};



//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//std::default_random_engine generator(seed);
//std::normal_distribution<float> dists[OUTPUT_DIM] = {};



GLFWwindow* setupImGui()
{
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        return nullptr;

#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    GLFWwindow* window = glfwCreateWindow(1800, 1000, "KalmanFilterImpl Project", NULL, NULL);
    if (window == nullptr)
        return nullptr;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigDockingWithShift = false;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;

    return window;
}

void imGuiNewFrame()
{
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void imGuiEndOfFrame(GLFWwindow* window)
{
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(CLEAR_COLOUR.x * CLEAR_COLOUR.w, CLEAR_COLOUR.y * CLEAR_COLOUR.w, CLEAR_COLOUR.z * CLEAR_COLOUR.w, CLEAR_COLOUR.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    GLFWwindow* backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);

    glfwSwapBuffers(window);
}

void imGuiDestroy(GLFWwindow* window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();
    ImPlot::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}


// --------- Sample systems ---------

// Simple spring-mass system
class : public SystemImpl<2, 1, 1>
{
    void setupFilter()
    {
        constexpr float SPRING_STIFFNESS = 0.1f; // N/m
        constexpr float MASS = 10.0f; // kg

        constexpr float TIMESTEP = 1.0f / 60.0f;

        systemMat << 1, TIMESTEP, -TIMESTEP * SPRING_STIFFNESS / MASS, 1;
        inputMat << 0, TIMESTEP* SPRING_STIFFNESS / MASS;
        outputMat << 1, 0;
        feedthroughMat << 0;
        processNoiseCov << 1e-10, 0, 0, 1e-10;
        measurementNoiseCov << 0.001;

        initialControlVec << 100;
    }

} simpleSpringMassSystem;



constexpr size_t STATE_DIM = 2, OUTPUT_DIM = 1, CONTROL_DIM = 1;
SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>* systemImpl = &simpleSpringMassSystem;


int main()
{

    GLFWwindow* window = setupImGui();
    
    if (window == nullptr)
        return 1;

    bool imgui_show_demo_window = false;
    bool implot_show_demo_window = false;

    systemImpl->createFilter();

    //dists[0] = std::normal_distribution<float>(0, std::sqrt(measurementNoiseCov(0, 0)));

    while (!glfwWindowShouldClose(window))
    {

        imGuiNewFrame();

        if (imgui_show_demo_window)
            ImGui::ShowDemoWindow(&imgui_show_demo_window);

        if (implot_show_demo_window)
            ImPlot::ShowDemoWindow(&implot_show_demo_window);

        //
        static Vector<CONTROL_DIM> controlVec = systemImpl->initialControlVec;
        const Vector<STATE_DIM> actualStateVec = systemImpl->updateAndGetActualState(controlVec);

        Vector<OUTPUT_DIM> measurementVec = systemImpl->getMeasurement(controlVec);
        KalmanFilterImpl::Gaussian<STATE_DIM> currentEstimate = systemImpl->getPrediction(controlVec, measurementVec);
        Vector<STATE_DIM> estimateMean = currentEstimate.getMean();
        controlVec(0) = 0;

        static float history = 10.0f;

        // Main window
        {
            ImGui::Begin("Visual");

            static ScrollingBuffer plotDataRef, plotDataMeas, plotDataFilt;

            static float t = 0;
            t += ImGui::GetIO().DeltaTime;

            plotDataRef.AddPoint(t, actualStateVec(0));
            plotDataMeas.AddPoint(t, measurementVec(0));
            plotDataFilt.AddPoint(t, estimateMean(0));

            if (ImPlot::BeginPlot("Data", ImVec2(-1, 900)))
            {
                static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels;
                ImPlot::SetupAxes(NULL, NULL, flags, flags);
                ImPlot::SetupAxisLimits(ImAxis_X1, t - history * 0.9, t + history * 0.1, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);
                ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);

                // Plot measurement signal
                ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1.0f);
                ImPlot::PlotScatter("Raw measurement", &plotDataMeas.Data[0].x, &plotDataMeas.Data[0].y, plotDataMeas.Data.size(), 0, plotDataMeas.m_offset, 2 * sizeof(float));
                ImPlot::PopStyleVar();

                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2);
                // Plot filtered signal
                ImPlot::PlotLine("Filtered measurement", &plotDataFilt.Data[0].x, &plotDataFilt.Data[0].y, plotDataFilt.Data.size(), 0, plotDataFilt.m_offset, 2 * sizeof(float));

                // Plot reference signal
                ImPlot::PlotLine("Reference", &plotDataRef.Data[0].x, &plotDataRef.Data[0].y, plotDataRef.Data.size(), 0, plotDataRef.m_offset, 2 * sizeof(float));
                ImPlot::PopStyleVar();

                ImPlot::EndPlot();
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Options");

            ImGui::DragFloat("History length", &history);

            ImGui::End();
        }

        imGuiEndOfFrame(window);
    }

    imGuiDestroy(window);

    delete systemImpl;

    return 0;
}
