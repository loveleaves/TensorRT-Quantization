cmake_minimum_required(VERSION 3.12)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(yolov8 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

# add common module
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(calibrate
    calibrate.cu
    calibrator.cpp
    yolov8_utils.cpp
)
target_link_libraries(calibrate PUBLIC ${PROJECT_BINARY_DIR} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(calibrate PUBLIC ${PROJECT_BINARY_DIR} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})