find_package(OpenMP)

add_library(common INTERFACE)
target_compile_options(common INTERFACE
	-Wall -Wextra -Wpedantic -Werror)

include_directories("../include/")

add_executable(quickstart.bin quickstart.cpp)
target_link_libraries(quickstart.bin PRIVATE common)

add_executable(sift_test.bin sift_test.cpp)
target_link_libraries(sift_test.bin PRIVATE common)
if (OpenMP_FOUND)
	target_link_libraries(sift_test.bin PRIVATE OpenMP::OpenMP_CXX)
endif()