add_custom_target(recoil_examples)

set(RECOIL_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

function(recoil_example_add_executable NAME)
    cmake_parse_arguments(_ "" "" "" ${ARGN})

    add_executable(${NAME} ${__UNPARSED_ARGUMENTS})

    add_dependencies(recoil_examples ${NAME})

    target_include_directories(
            ${NAME}
            PRIVATE
            ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}
    )
endfunction()

foreach(EXAMPLE
        00_encode_decode_textfile
        01_encode_decode_textfile_split
        02_encode_decode_textfile_split_cuda
        03_encode_decode_textfile_symbolsplit
        04_encode_decode_textfile_symbolsplit_cuda

        10_encode_textfile
        11_encode_textfile_symbolsplit)
    add_subdirectory(${EXAMPLE})
endforeach()