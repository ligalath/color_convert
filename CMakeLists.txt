CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
PROJECT(color_convert)

#import cuda
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA)
message("cuda_include: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

#import opencv
set(OpenCV_DIR /usr/local/opencv/share/OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(color_convert main.cpp color_convert.cu)
target_link_libraries(color_convert ${OpenCV_LIBS})