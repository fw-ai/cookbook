cmake_minimum_required(VERSION 3.11)
project(testdriver)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(NIST_EXTERN_BVT_STRUCTS_VERSION NIST_EXTERN_API_VERSION)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

add_executable(validate11 validate11.cpp)
target_link_libraries(validate11 $ENV{BVT_IMPL_LIB})
