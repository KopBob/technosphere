project(calculator_tests)

add_subdirectory(lib/gtest-1.7.0)
add_subdirectory(basic_tests)
add_subdirectory(calc_tests)

include_directories(lib/gtest-1.7.0)
