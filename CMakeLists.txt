cmake_minimum_required(VERSION 3.24)
project(recoil LANGUAGES CXX)

include(CheckCXXCompilerFlag)

find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(CMAKE_CUDA_FLAGS "-std=c++20")

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    set(CMAKE_CUDA_FLAGS_DEBUG "-G ${CMAKE_CUDA_FLAGS}")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -G ${CMAKE_CUDA_FLAGS}")
else()
    set(CUDA_INCLUDE_DIRS)
    set(CUDA_LIBRARIES)
    message(WARNING "CUDA examples cannot be compiled, since CUDA is not found on this system")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "-pthread")

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Winline -DNDEBUG ${CMAKE_CXX_FLAGS}")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        $ENV{INCLUDE_DIR}
)
link_directories(
        ${CUDA_LIBRARIES}
        $ENV{LINK_DIR}
)

add_executable(test test.cpp)

add_subdirectory(examples)