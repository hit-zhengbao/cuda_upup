#include "reduce.h"

// 该方法使用warp的shuffle 来实现，未参考前几个版本的实现借鉴的github作者实现

namespace cudaup
{
#define WARP_SIZE 32

CUDA_DEVICE int WarpReduce(int val)
{
    for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, i);
    }

    return val;
}

CUDA_GLOBAL void ReduceAllUseWarpKernel(Mat *mat, Mat *sum)
{
    int t_id     = threadIdx.x;
    int block_id = blockIdx.x;
    int g_id = block_id * blockDim.x + t_id;

    int val = g_id < mat->m_sizes.m_w ? mat->at<int32_t>(0, g_id) : 0;

    int lane = t_id % WARP_SIZE;    // 每个线程在warp中的id
    int w_id = t_id / WARP_SIZE;    // 每个线程属于哪个warp

    static __shared__ int s_sum[WARP_SIZE];

    // 1. 每个warp计算自己的sum
    val = WarpReduce(val);

    if (0 == lane)
    {
        s_sum[w_id] = val;  // warp内的归约结果
    }
    __syncthreads();

    // 第一个warp计算所有warp的sum
    if (0 == w_id)
    {
        val = t_id < (blockDim.x / WARP_SIZE) ? s_sum[lane] : 0;
        val = WarpReduce(val);

        if (0 == lane)
        {
            sum->at<int32_t>(0, block_id) = val;
        }
    }
}

int32_t ReduceAllUseWarp(Mat &mat, int32_t &sum)
{
    if (mat.empty())
    {
        LOG_ERROR("mat is empty");
        return RET_ERR;
    }

    if (mat.m_sizes.m_ch != 1)
    {
        LOG_ERROR("only support S32C1");
        return RET_ERR;
    }

    Mat sum_mat_gpu = mat.clone(MemType::MEM_GPU);

    const int32_t BLOCK_SIZE = 256; // 至少是warp(32)的整数倍

    dim3 block_size(BLOCK_SIZE);
    dim3 global_size(CEIL_DIV(mat.m_sizes.m_w, BLOCK_SIZE));

    ReduceAllUseWarpKernel<<<global_size, block_size>>>(mat.GetMatAllOnCUDAMem(), sum_mat_gpu.GetMatAllOnCUDAMem());

    Mat sum_mat_cpu = sum_mat_gpu.clone(MemType::MEM_CPU);

    sum = 0;
    for (int32_t i = 0; i < global_size.x; ++i)
    {
        sum += sum_mat_cpu.at<int32_t>(0, i);
    }

    return RET_OK;
}
} // namespace cudaup