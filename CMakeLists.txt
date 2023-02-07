cmake_minimum_required(VERSION 3.10)
project(recoil)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "")

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        $ENV{INCLUDE_DIR}
)
link_directories(
        $ENV{LINK_DIR}
)

add_executable(test test.cpp)
