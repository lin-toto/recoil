recoil_example_add_executable(
        encode_decode_textfile encode_decode_textfile.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/cdf_utils.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/file.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/profiling.cpp
)