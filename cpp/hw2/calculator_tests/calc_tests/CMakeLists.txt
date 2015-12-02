include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
endif (APPLE)

add_executable(runCalcTests calc_check.cpp)
target_link_libraries(runCalcTests calculator)
target_link_libraries(runCalcTests gtest gtest_main)

