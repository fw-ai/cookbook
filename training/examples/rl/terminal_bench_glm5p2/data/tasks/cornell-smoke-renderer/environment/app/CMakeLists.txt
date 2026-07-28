cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/main.cc)
target_include_directories(raytracer PRIVATE src)
