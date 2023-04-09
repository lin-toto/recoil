if(CUDA_FOUND)
    recoil_example_add_executable(
        detect_cuda_occupancy detect_cuda_occupancy.cu
    )
endif()