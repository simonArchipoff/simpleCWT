cmake_minimum_required(VERSION 3.5)

project(simpleCWT LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/fCWT_wrapper)

add_executable(simpleCWT
    main.cpp
    simplecwt.h
    simplecwt.cpp)

find_package(PkgConfig)
pkg_search_module(FFTW REQUIRED fftw3 fftw3_omp IMPORTED_TARGET)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simpleCWT PUBLIC OpenMP::OpenMP_CXX)
endif()
if(true)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -funsafe-math-optimizations")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
endif()
target_link_libraries(simpleCWT PUBLIC fftw3f fftw3_omp fCWT)
