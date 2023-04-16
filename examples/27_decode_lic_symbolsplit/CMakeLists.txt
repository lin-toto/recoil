recoil_example_add_executable(
        decode_lic_symbolsplit decode_lic_symbolsplit.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/cdf_utils.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/file.cpp
        ${RECOIL_EXAMPLES_COMMON_SOURCE_DIR}/profiling.cpp
)