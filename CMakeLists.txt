cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

project(CUDAUp LANGUAGES CXX CUDA)

# Set c++ stanard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INC_DIRS "" CACHE STRING "include files")
set(TEST_SRC "" CACHE STRING "src impl files")

find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif()

add_subdirectory(src)

add_subdirectory(test)
