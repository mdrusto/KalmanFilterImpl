project(imgui LANGUAGES C CXX)

find_package(OpenGL REQUIRED)
#find_package(glfw REQUIRED)
#find_package(glad REQUIRED)
#find_package(glm REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_library(imgui STATIC)

target_sources(
		imgui
		PRIVATE
			${IMGUI_DIR}/imgui_demo.cpp
			${IMGUI_DIR}/imgui_draw.cpp
			${IMGUI_DIR}/imgui_tables.cpp
			${IMGUI_DIR}/imgui_widgets.cpp
			${IMGUI_DIR}/imgui.cpp
		
		PRIVATE
			${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
			${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
		)

target_include_directories(
		imgui
		PUBLIC
			${IMGUI_DIR}
			${IMGUI_DIR}/backends
		)

target_link_libraries(imgui PUBLIC ${OPENGL_LIBRARIES} glfw)
