macro(add_new_module module)
    # Header files
    list(APPEND INC_CUR_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${module}/inc")
    list(APPEND INC_CUR_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${module}/priv")

    # Implement sources
    aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/${module}/src" IMPL_CUR_SRC)

    # test sources
    aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/${module}/test" TEST_CUR_SRC)

endmacro()

add_new_module(simple_gtest)
add_new_module(matmuls)
add_new_module(utils)
add_new_module(reduce)

include_directories("${INC_CUR_DIRS}")
message(STATUS "INC: ${INC_CUR_DIRS}")

set(INC_DIRS "${INC_CUR_DIRS}" CACHE STRING "update include files" FORCE)
set(TEST_SRC "${TEST_CUR_SRC}" CACHE STRING "update src impl files" FORCE)

message(STATUS "IMPL_CUR_SRC: ${IMPL_CUR_SRC}")

add_library(CUDAUp STATIC ${IMPL_CUR_SRC})
set_target_properties(CUDAUp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # 支持分离编译
    CUDA_STANDARD 17               # CUDA 使用 C++14 标准
)

target_link_libraries(CUDAUp PRIVATE CUDA::cudart)

install(TARGETS CUDAUp DESTINATION lib)