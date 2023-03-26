if(CUDA_FOUND)
    recoil_example_add_executable(
        encode_decode_textfile_split_cuda encode_decode_textfile_split.cu
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/cdf_utils.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/file.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/profiling.cpp
    )
endif()