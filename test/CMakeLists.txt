include_directories("${INC_DIRS}")
message(STATUS "INC: ${INC_DIRS}")

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TEST_SRC)
message(STATUS "TEST_SRC: ${TEST_SRC}")

add_executable(CUDAUpExe ${TEST_SRC})

target_link_libraries(CUDAUpExe CUDAUp CUDA::cudart)

set_target_properties(CUDAUpExe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # 支持分离编译
    CUDA_STANDARD 17               # CUDA 使用 C++14 标准
)

install(TARGETS CUDAUpExe  DESTINATION test)