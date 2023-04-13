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
        10_encode_textfile
        11_encode_textfile_symbolsplit

        12_combine_encoded_splits
        13_detect_cuda_occupancy

        15_decode_textfile_split
        16_decode_textfile_split_cuda
        17_decode_textfile_symbolsplit
        18_decode_textfile_symbolsplit_cuda

        20_encode_lic)
    add_subdirectory(${EXAMPLE})
endforeach()