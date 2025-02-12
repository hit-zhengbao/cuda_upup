#include "reduce.h"

// 参考实现： https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/01_interleaved_addressing/README.md

namespace cudaup
{
template<int32_t BLOCK_SIZE>
CUDA_GLOBAL void ReduceInterLeaveAddrKernel(Mat *mat, Mat *sum)
{
    __shared__ int32_t shared_data[BLOCK_SIZE];

    int32_t t_id  = threadIdx.x;
    int32_t b_id  = blockIdx.x;
    int32_t b_dim = blockDim.x;
    int32_t g_id  = b_dim * b_id + t_id;

    if (g_id >= mat->m_sizes.m_w)
    {
        return;
    }

    shared_data[t_id] = mat->at<int32_t>(0, g_id);

    __syncthreads();

    // Use interleaved addressing
    for (int32_t s = 1; s < BLOCK_SIZE; s *= 2)
    {
        int32_t index = 2 * s * t_id;

        if (index + s < BLOCK_SIZE && (b_dim * b_id + s < mat->m_sizes.m_w))
        {
            shared_data[index] += shared_data[index + s];
        }

        __syncthreads();
    }

    if (t_id == 0)
    {
        sum->at<int32_t>(0, b_id) = shared_data[0];
    }
}

int32_t ReduceInterLeaveAddr(Mat &mat, int32_t &sum)
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

    ReduceInterLeaveAddrKernel<BLOCK_SIZE><<<global_size, block_size>>>(mat.GetMatAllOnCUDAMem(), sum_mat_gpu.GetMatAllOnCUDAMem());

    Mat sum_mat_cpu = sum_mat_gpu.clone(MemType::MEM_CPU);

    sum = 0;
    for (int32_t i = 0; i < global_size.x; ++i)
    {
        sum += sum_mat_cpu.at<int32_t>(0, i);
    }

    return RET_OK;
}
} // namespace cudaup