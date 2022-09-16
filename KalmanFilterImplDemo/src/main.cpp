#include "KalmanFilterLib/Gaussian.h"
#include "KalmanFilterLib/ExtendedKalmanFilter.h"

#include "SystemImpl.h"
#include "LinearSystem.h"
#include "NonlinearSystem.h"

//#include <chrono>

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
    //io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
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
constexpr float SPRING_STIFFNESS = 10.0f; // N/m
constexpr float MASS = 10.0f; // kg
LinearSystem<2, 1, 1> simpleSpringMassSystem(
        (Matrix<2, 2>() << 0, 1, -SPRING_STIFFNESS / MASS, 0).finished(),
        (Matrix<2, 1>() << 0, 1 / MASS).finished(),
        (Matrix<1, 2>() << 1, 0).finished(),
        Matrix<1, 1>(),
        1e-10 * Matrix<2, 2>::Identity(),
        0.0001 * Matrix<1, 1>::Identity());


constexpr float RADIUS = 0.1f;
constexpr float I_x = 1.0f, I_y = 1.0f, I_z = 1.0f;
constexpr float FM_SCALING_FACTOR = 1.0f;
LinearSystem<6, 3, 4> linearQuadcopter3DOF(
        (Matrix<6, 6>() <<
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0).finished(),
        (Matrix<6, 4>() <<
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, RADIUS / I_x, 0, -RADIUS / I_x,
            RADIUS / I_y, 0, -RADIUS / I_y, 0,
            -FM_SCALING_FACTOR / I_z, FM_SCALING_FACTOR / I_z, -FM_SCALING_FACTOR / I_z, FM_SCALING_FACTOR / I_z).finished(),
        (Matrix<3, 6>() <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0).finished(), 
        Matrix<3, 4>(), 
        1.0f * Matrix<6, 6>::Identity(), 
        1.0f * Matrix<3, 3>::Identity());

constexpr float G = 9.81f; // m/s^2
constexpr float QC_MASS = 1.0f; // kg
LinearSystem<12, 6, 4> linearQuadcopter6DOF(
    (Matrix<12, 12>() <<
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, -G, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, G, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
    (Matrix<12, 4>() <<
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1/QC_MASS, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 1 / I_x, 0, 0,
        0, 0, 1 / I_y, 0,
        0, 0, 0, 1 / I_z).finished(),
        (Matrix<6, 12>() <<
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0).finished(),
    Matrix<6, 4>(),
    0.001f * Matrix<12, 12>::Identity(),
    1.0e-6f * Matrix<6, 6>::Identity());


//NonlinearSystem<12, 6, 4> nonlinearQuadcopter6DOF(

//);


// Select system to use here
constexpr size_t STATE_DIM = 12, OUTPUT_DIM = 6, CONTROL_DIM = 4;
SystemImpl<STATE_DIM, OUTPUT_DIM, CONTROL_DIM>* systemImpl = &linearQuadcopter6DOF;


// GUI helper functions
template <size_t DIM>
void displayVector(Vector<DIM> vector) {
    if (ImGui::BeginTable("Vector", 1, ImGuiTableFlags_Borders | ImGuiTableFlags_NoHostExtendX)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        for (size_t i = 0; i < DIM; i++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%.4f", vector(i));
        }
        ImGui::EndTable();
    }
}

template <size_t ROWS, size_t COLS>
void displayMatrix(Matrix<ROWS, COLS> matrix, float colWidth = 100.0f) {
    static int callCount = 0;
    if (ImGui::BeginTable(("Table #" + std::to_string(callCount++)).c_str(), COLS, ImGuiTableFlags_Borders)) {
        for (int j = 0; j < COLS; j++) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, colWidth);
        }
        for (size_t i = 0; i < ROWS; i++) {
            ImGui::TableNextRow();
            for (size_t j = 0; j < COLS; j++) {
                ImGui::TableSetColumnIndex(j);
                ImGui::Text("%.4f", matrix(i, j));
            }
        }
        ImGui::EndTable();
    }
}

int main()
{

    GLFWwindow* window = setupImGui();
    
    if (window == nullptr)
        return 1;

    bool imgui_show_demo_window = false;
    bool implot_show_demo_window = false;

    systemImpl->initSystem();

    // Update system once before loop starts so we can start with non-zero delta time
    systemImpl->updateAndGetActualState(Vector<CONTROL_DIM>(0));
    systemImpl->getPrediction(Vector<CONTROL_DIM>(0), Vector<OUTPUT_DIM>(0));

    while (!glfwWindowShouldClose(window))
    {

        imGuiNewFrame();

        if (imgui_show_demo_window)
            ImGui::ShowDemoWindow(&imgui_show_demo_window);

        if (implot_show_demo_window)
            ImPlot::ShowDemoWindow(&implot_show_demo_window);

        static Vector<CONTROL_DIM> controlVec = Vector<CONTROL_DIM>(0, 0, 0, 0);

        static float inputThrust = 0.0f;
        controlVec(0) = inputThrust;

        const Vector<STATE_DIM> actualStateVec = systemImpl->updateAndGetActualState(controlVec);
        //std::cout << actualStateVec << std::endl;

        Vector<OUTPUT_DIM> measurementVec = systemImpl->calculateMeasurement(controlVec);

        auto timeBefore = std::chrono::high_resolution_clock::now();
        KalmanFilterImpl::Gaussian<STATE_DIM> currentEstimate = systemImpl->getPrediction(controlVec, measurementVec);
        auto timeAfter = std::chrono::high_resolution_clock::now();

        Vector<STATE_DIM> estimateMean = currentEstimate.getMean();

        std::chrono::nanoseconds durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(timeAfter - timeBefore);
        const float executionTimeUs = durationNs.count() * 1.0e-3;

        static float history = 10.0f;

        ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_None;

        ImGuiWindowFlags dockWindowFlags = ImGuiWindowFlags_NoDocking;
        dockWindowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        dockWindowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        static bool dockspaceOpen = true;

        ImGuiViewport* mainViewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(mainViewport->Pos);
        ImGui::SetNextWindowSize(mainViewport->Size);
        ImGui::SetNextWindowViewport(mainViewport->ID);

        ImGui::Begin("Dockspace window", &dockspaceOpen, dockWindowFlags);

        ImGuiID dockspaceID = ImGui::GetID("MyDockspace");
        ImGui::DockSpace(dockspaceID, {0, 0}, dockspaceFlags);

        // Main window
        
        if (ImGui::Begin("Visual"))
        {
            static ScrollingBuffer plotDataRef, plotDataMeas, plotDataFilt;

            static float t = 0;
            t += ImGui::GetIO().DeltaTime;

            plotDataRef.AddPoint(t, actualStateVec(2));
            plotDataMeas.AddPoint(t, measurementVec(2));
            plotDataFilt.AddPoint(t, estimateMean(2));

            if (ImPlot::BeginPlot("Data", ImGui::GetContentRegionAvail()))
            {
                static ImPlotAxisFlags flags = ImPlotAxisFlags_None;
                ImPlot::SetupAxes("Time (s)", "Position (m)", flags, flags);
                ImPlot::SetupAxisLimits(ImAxis_X1, t - history * 0.9, t + history * 0.1, ImGuiCond_Always);
                const static Vector<STATE_DIM> initialState = systemImpl->m_currentState;
                const static float yLim = initialState(0) * 1.1f;
                ImPlot::SetupAxisLimits(ImAxis_Y1, -yLim, yLim);
                ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);

                // Plot measurement signal
                ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1.0f);
                ImPlot::PlotLine("Raw measurement", &plotDataMeas.Data[0].x, &plotDataMeas.Data[0].y, plotDataMeas.Data.size(), 0, plotDataMeas.m_offset, 2 * sizeof(float));
                ImPlot::PopStyleVar();

                // Plot reference signal
                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2);
                ImPlot::PlotLine("Reference (true value)", &plotDataRef.Data[0].x, &plotDataRef.Data[0].y, plotDataRef.Data.size(), 0, plotDataRef.m_offset, 2 * sizeof(float));

                // Plot filtered signal
                ImPlot::PlotLine("Filtered measurement", &plotDataFilt.Data[0].x, &plotDataFilt.Data[0].y, plotDataFilt.Data.size(), 0, plotDataFilt.m_offset, 2 * sizeof(float));
                ImPlot::PopStyleVar();

                ImPlot::EndPlot();
            }

            ImGui::End();
        }

        ImGui::SetNextWindowSizeConstraints({ 60, 300 }, { 60, -1 });
        if (ImGui::Begin("Control"))
        {
            ImGui::VSliderFloat("##", ImGui::GetContentRegionAvail(), &inputThrust, -10.0f, 10.0f);

            ImGui::End();
        }

        if (ImGui::Begin("Options"))
        {
            ImGui::DragFloat("History length", &history);

            char execTimeBuffer[30];
            sprintf(execTimeBuffer, "Execution time: %.2f us", executionTimeUs);
            //std::cout << execTimeBuffer << std::endl;
            ImGui::Text(execTimeBuffer);

            char deltaTimeBuffer[30];
            sprintf(deltaTimeBuffer, "Delta time: %.2f ms", systemImpl->m_currentDeltaTime * 1e3f);
            ImGui::Text(deltaTimeBuffer);

            ImGui::End();
        }

        if (ImGui::Begin("Parameters"))
        {
            //ImGui::Text("System matrix");
            //displayMatrix<STATE_DIM, STATE_DIM>(systemImpl->m_systemMat, 40.0f);
            //
            //ImGui::Text("Input matrix");
            //displayMatrix<STATE_DIM, CONTROL_DIM>(systemImpl->m_inputMat, 40.0f);
            //
            //ImGui::Text("Output matrix");
            //displayMatrix<OUTPUT_DIM, STATE_DIM>(systemImpl->m_outputMat, 40.0f);
            //
            //ImGui::Text("Feedthrough matrix");
            //displayMatrix<OUTPUT_DIM, CONTROL_DIM>(systemImpl->m_feedthroughMat, 40.0f);

            ImGui::Text("Process noise covariance");
            displayMatrix<STATE_DIM, STATE_DIM>(systemImpl->m_processNoiseCov, 40.0f);

            ImGui::Text("Measurement noise covariance");
            displayMatrix<OUTPUT_DIM, OUTPUT_DIM>(systemImpl->m_measurementNoiseCov, 40.0f);

            ImGui::End();
        }

        ImGui::End();

        imGuiEndOfFrame(window);
    }

    imGuiDestroy(window);

    delete systemImpl;

    return 0;
}
