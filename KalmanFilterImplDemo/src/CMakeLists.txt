include_directories(${KalmanFilterImpl_SOURCE_DIR}/KalmanFilterImplDemo/include)

include_directories(${KalmanFilterImpl_SOURCE_DIR}/KalmanFilterLib/include)

add_executable(KalmanFilterImplDemo main.cpp)
target_link_libraries(KalmanFilterImplDemo KalmanFilterLib imgui implot)
