cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)

project(embdb LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR})

add_library(common INTERFACE)

target_compile_options(common INTERFACE
	-Wall -Wextra -Wpedantic -Werror)

add_executable(embdb.bin src/embdb.cpp)
target_link_libraries(embdb.bin PRIVATE common)
