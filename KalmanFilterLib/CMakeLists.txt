add_library(KalmanFilterLib INTERFACE)

target_link_libraries(KalmanFilterLib INTERFACE Eigen3::Eigen)

target_include_directories(KalmanFilterLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
