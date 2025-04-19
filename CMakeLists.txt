cmake_minimum_required(VERSION 3.10)
project(MarchingSquares)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(marching_squares marching_squares.cpp)
target_link_libraries(marching_squares ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
