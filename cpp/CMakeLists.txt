cmake_minimum_required(VERSION 3.18...3.30)
project(g2s_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(g2s_cpp NB_STATIC src/bindings.cpp)
target_include_directories(g2s_cpp PRIVATE src)
if(MSVC)
  target_compile_options(g2s_cpp PRIVATE /O2 /W3)
else()
  target_compile_options(g2s_cpp PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS g2s_cpp LIBRARY DESTINATION .)
