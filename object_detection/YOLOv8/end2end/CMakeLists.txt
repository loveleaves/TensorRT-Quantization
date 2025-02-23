cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(Function)

# 指定 CUDA 路径（根据你的安装路径来设置）
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

# CUDA
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
find_package(OpenCV REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
message(STATUS "TensorRT_INCLUDE_DIRS: " ${TensorRT_INCLUDE_DIRS})
print_var(TensorRT_LIB_DIR)
if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)
print_var(ALL_LIB_DIRS)

# add common module
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(
        ${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        ${ALL_LIBS}
)
