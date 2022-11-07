cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

project(embdb LANGUAGES CXX)

find_package(spdlog REQUIRED)

add_library(common INTERFACE)
target_compile_options(common INTERFACE
	-Wall -Wextra -Wpedantic -Werror)
target_link_libraries(common INTERFACE spdlog::spdlog)

include_directories("${CMAKE_SOURCE_DIR}/include/")

add_subdirectory(examples)