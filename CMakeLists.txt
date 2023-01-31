cmake_minimum_required(VERSION 3.20)
project(recoil)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "")

set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")

include_directories(
        ${CMAKE_SOURCE_DIR}
        $ENV{INCLUDE_DIR}
)
link_directories(
        $ENV{LINK_DIR}
)

add_executable(test test.cpp)
