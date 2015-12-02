cmake_minimum_required(VERSION 2.8)

project(calculator)

set(HEADER_FILES
    calc.h)

set(SOURCE_FILES
    calc.c)

add_library(calculator STATIC ${SOURCE_FILES} ${HEADER_FILES})
