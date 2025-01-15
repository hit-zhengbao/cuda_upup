#include "simple_gtest.h"
#include "mat_mul.h"
#include "mat.h"
#include "def.h"
#include "timer.h"

#include <functional>

struct MatTestCase
{
    using FuncType = int32_t(*)(cudaup::Mat &, cudaup::Mat &, cudaup::Mat &);

    cudaup::Sizes sizes;

    FuncType impl_func;
    FuncType ref_func;
    std::string name;

    MatTestCase(cudaup::Sizes sizes, FuncType impl_func, FuncType ref_func, std::string name)
        : sizes(sizes), impl_func(impl_func), ref_func(ref_func), name(name)
    {}
};

static int32_t MatMulTestImpl(const MatTestCase& test_case)
{
    int32_t ret = RET_ERR;

    // 1. Generate random data
    const cudaup::Sizes &sizes = test_case.sizes;
    const cudaup::Sizes strides{ALIGN(sizes.m_w * sizes.m_ch * (int32_t)sizeof(float), 128)};

    cudaup::Mat a_cpu(sizes, cudaup::MatType::MAT_F32, cudaup::MemType::MEM_CPU, strides);
    cudaup::Mat b_cpu(sizes, cudaup::MatType::MAT_F32, cudaup::MemType::MEM_CPU, strides);
    cudaup::Mat c_cpu(sizes, cudaup::MatType::MAT_F32, cudaup::MemType::MEM_CPU, strides);

    // Generate random data
    a_cpu.random<float>();
    b_cpu.random<float>();

    cudaup::Mat a_gpu = a_cpu.clone(cudaup::MemType::MEM_GPU);
    cudaup::Mat b_gpu = b_cpu.clone(cudaup::MemType::MEM_GPU);
    cudaup::Mat c_gpu = c_cpu.clone(cudaup::MemType::MEM_GPU);

    if (a_cpu.empty() || b_cpu.empty() || c_cpu.empty() || a_gpu.empty() || b_gpu.empty() || c_gpu.empty())
    {
        LOG_ERROR("MatMulTestImpl: mat is empty!");
        return ret;
    }

    // 2. Call GPU impl function
    {
        std::string tag_name = test_case.name + " " + std::to_string(sizes.m_w) + " * " + std::to_string(sizes.m_h) + " * " + std::to_string(sizes.m_ch);
        cudaup::TimeTracker time_tracker(tag_name);

        ret = test_case.impl_func(a_gpu, b_gpu, c_gpu);
        if (ret != RET_OK)
        {
            LOG_ERROR("MatMulTestImpl: call impl function failed!");
            return ret;
        }
    }
    
    // 3. Check accuracy
    ret = test_case.ref_func(a_cpu, b_cpu, c_cpu);
    if (ret != RET_OK)
    {
        LOG_ERROR("MatMulTestImpl: call ref function failed!");
        return ret;
    }

    cudaup::Mat c_gpu_cpu = c_gpu.clone(cudaup::MemType::MEM_CPU);
    if (c_gpu_cpu.empty())
    {
        LOG_ERROR("MatMulTestImpl: c_gpu_cpu is empty!");
        return ret;
    }

    ret = c_cpu.compare<float>(c_gpu_cpu);
    if (ret != RET_OK)
    {
        LOG_ERROR("MatMulTestImpl: c_cpu and c_gpu_cpu are not equal!");
        return ret;
    }

    ret = RET_OK;
    return ret;
}

SIMPLE_TEST(mat_mul_test)
{
    std::vector<MatTestCase> test_cases = {
        {{255,   255, 1}, cudaup::MatMulBlockAndNoLocalMem, cudaup::MatMulScalar, "MatMulBlockAndNoLocalMem"},
        {{256,   256, 1}, cudaup::MatMulBlockAndNoLocalMem, cudaup::MatMulScalar, "MatMulBlockAndNoLocalMem"},
        {{512,   512, 1}, cudaup::MatMulBlockAndNoLocalMem, cudaup::MatMulScalar, "MatMulBlockAndNoLocalMem"},
        {{1024, 1024, 1}, cudaup::MatMulBlockAndNoLocalMem, cudaup::MatMulScalar, "MatMulBlockAndNoLocalMem"},
        {{2048, 2048, 1}, cudaup::MatMulBlockAndNoLocalMem, cudaup::MatMulScalar, "MatMulBlockAndNoLocalMem"}
    };

    for (const auto &test_case : test_cases)
    {
        MatMulTestImpl(test_case);
    }
}