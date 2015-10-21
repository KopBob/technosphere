cmake_minimum_required (VERSION 3.3)
project (SphereProblem1)

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -O2 -std=c99 -lm")

set(SOURCE_FILES main.c)
add_executable(SphereProblem1 ${SOURCE_FILES})