cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)

project(embdb LANGUAGES CXX)

find_package(OpenMP)

include_directories(${CMAKE_SOURCE_DIR})

add_library(common INTERFACE)

target_compile_options(common INTERFACE
	-Wall -Wextra -Wpedantic -Werror)

add_executable(embdb.bin src/embdb.cpp)
target_link_libraries(embdb.bin PRIVATE common)

add_executable(sift_test.bin src/sift_test.cpp)
target_link_libraries(sift_test.bin PRIVATE common)
if (OpenMP_FOUND)
	target_link_libraries(sift_test.bin PRIVATE OpenMP::OpenMP_CXX)
endif()