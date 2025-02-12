#include "simple_gtest.h"
#include "reduce.h"
#include "mat.h"
#include "def.h"
#include "timer.h"

struct ReduceTestCase
{
    using FuncType = int32_t(*)(cudaup::Mat &, int32_t &);

    cudaup::Sizes sizes;

    FuncType impl_func;
    FuncType ref_func;
    std::string name;

    ReduceTestCase(cudaup::Sizes sizes, FuncType impl_func, FuncType ref_func, std::string name)
        : sizes(sizes), impl_func(impl_func), ref_func(ref_func), name(name)
    {}
};

static int32_t ReduceTestImpl(const ReduceTestCase& test_case)
{
    int32_t ret = RET_ERR;

    // 1. Generate random data
    const cudaup::Sizes &sizes = test_case.sizes;
    const cudaup::Sizes strides{ALIGN(sizes.m_w * sizes.m_ch * (int32_t)sizeof(int32_t), 128)};

    cudaup::Mat a_cpu(sizes, cudaup::MatType::MAT_S32, cudaup::MemType::MEM_CPU, strides);
    int32_t sum_cpu = 0, sum_gpu = 0;

    // Generate random data
    a_cpu.random<int32_t>();

    cudaup::Mat a_gpu = a_cpu.clone(cudaup::MemType::MEM_GPU);

    if (a_cpu.empty() || a_gpu.empty())
    {
        LOG_ERROR("ReduceTestImpl: mat is empty!");
        return ret;
    }

    LOG_INFO("ReduceTestImpl: %s, (%d*%d*%d)", test_case.name.c_str(), sizes.m_w, sizes.m_h, sizes.m_ch);

    // 2. Call GPU impl function
    {
        std::string tag_name = test_case.name + " " + std::to_string(sizes.m_w) + " * " + std::to_string(sizes.m_h) + " * " + std::to_string(sizes.m_ch);
        cudaup::TimeTracker time_tracker(tag_name);

        ret = AnalyzeFuncTime(10, 2, test_case.impl_func, std::ref(a_gpu), std::ref(sum_gpu));
        if (ret != RET_OK)
        {
            LOG_ERROR("ReduceTestImpl: call impl function failed!");
            return ret;
        }
    }
    
    // 3. Check accuracy
    ret = test_case.ref_func(a_cpu, sum_cpu);
    if (ret != RET_OK)
    {
        LOG_ERROR("ReduceTestImpl: call ref function failed!");
        return ret;
    }

    if (sum_cpu != sum_gpu)
    {
        LOG_ERROR("ReduceTestImpl: sum_cpu and sum_gpu are not equal! sum_cpu: %d, sum_gpu: %d\n", sum_cpu, sum_gpu);
        return ret;
    }
    else
    {
        LOG_INFO("ReduceTestImpl: compare ok!\n");
    }

    ret = RET_OK;
    return ret;
}

SIMPLE_TEST(reduce_test)
{
    std::vector<ReduceTestCase> test_cases = {
        // 1. ******* ReduceNative ********
        {{1024, 1, 1}, cudaup::ReduceNative, cudaup::ReduceScalar, "ReduceNative"},
        {{10000, 1, 1}, cudaup::ReduceNative, cudaup::ReduceScalar, "ReduceNative"},
        {{32 * 1024 * 1024, 1, 1}, cudaup::ReduceNative, cudaup::ReduceScalar, "ReduceNative"},

        // 2. ******* ReduceInterLeaveAddr ********
        {{1024, 1, 1},              cudaup::ReduceInterLeaveAddr, cudaup::ReduceScalar, "ReduceInterLeaveAddr"},
        {{10000, 1, 1},             cudaup::ReduceInterLeaveAddr, cudaup::ReduceScalar, "ReduceInterLeaveAddr"},
        {{32 * 1024 * 1024, 1, 1},  cudaup::ReduceInterLeaveAddr, cudaup::ReduceScalar, "ReduceInterLeaveAddr"},
    };

    for (const auto &test_case : test_cases)
    {
        ReduceTestImpl(test_case);
    }
}