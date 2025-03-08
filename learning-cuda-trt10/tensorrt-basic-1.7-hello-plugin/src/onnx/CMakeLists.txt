# 让 CMake 自动查找所有 .cpp 文件
file(GLOB A_SOURCES "*.cpp")

# 创建静态库或共享库
add_library(onnx_lib ${A_SOURCES})

# 指定头文件路径
target_include_directories(onnx_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
