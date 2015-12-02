cmake_minimum_required(VERSION 3.3)
project(hw2)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")

set(SOURCE_FILES main.c)
add_executable(hw2 ${SOURCE_FILES})

include_directories(calculator)

add_subdirectory(calculator)
add_subdirectory(calculator_tests)

target_link_libraries(hw2 calculator)