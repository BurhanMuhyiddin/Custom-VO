cmake_minimum_required(VERSION 3.10)

project(CustomVO)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
    src/dataset_handler.cpp
)

link_libraries(stdc++fs)

add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} stdc++fs)