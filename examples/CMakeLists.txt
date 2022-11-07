add_executable(quickstart.bin quickstart.cpp)
target_link_libraries(quickstart.bin PRIVATE common)

add_executable(sift_test.bin sift_test.cpp)
target_link_libraries(sift_test.bin PRIVATE common)

add_executable(integers.bin integers.cpp)
target_link_libraries(integers.bin PRIVATE common)

find_package(OpenMP)

if (OpenMP_FOUND)
	target_link_libraries(sift_test.bin PRIVATE OpenMP::OpenMP_CXX)
	target_link_libraries(integers.bin PRIVATE OpenMP::OpenMP_CXX)
endif()