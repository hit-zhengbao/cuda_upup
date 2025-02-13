#include "reduce.h"

namespace cudaup
{
CUDA_DEVICE void warp_reduce(volatile int32_t *shared_data, int32_t t_id)
{
    shared_data[t_id] += shared_data[t_id + 32];
    shared_data[t_id] += shared_data[t_id + 16];
    shared_data[t_id] += shared_data[t_id + 8];
    shared_data[t_id] += shared_data[t_id + 4];
    shared_data[t_id] += shared_data[t_id + 2];
    shared_data[t_id] += shared_data[t_id + 1];
}

template<int32_t BLOCK_SIZE>
CUDA_GLOBAL void ReduceUnrollLastWarpKernel(Mat *mat, Mat *sum)
{
    __shared__ int32_t shared_data[BLOCK_SIZE];

    int32_t t_id  = threadIdx.x;
    int32_t b_id  = blockIdx.x;
    int32_t b_dim = blockDim.x;
    int32_t g_id  = b_dim * 2 * b_id + t_id;

    if (g_id >= mat->m_sizes.m_w)
    {
        return;
    }

    shared_data[t_id] = mat->at<int32_t>(0, g_id) + mat->at<int32_t>(0, g_id + b_dim);

    __syncthreads();

    // Use bank conflict free
    for (int32_t s = (b_dim >> 1); s > 32; (s >>= 1))
    {
        if (t_id < s)
        {
            shared_data[t_id] += shared_data[t_id + s];
        }

        __syncthreads();
    }

    if (t_id < 32)
    {
        warp_reduce(shared_data, t_id);
    }

    if (t_id == 0)
    {
        sum->at<int32_t>(0, b_id) = shared_data[0];
    }
}

int32_t ReduceUnrollLastWarpFree(Mat &mat, int32_t &sum)
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
    dim3 global_size(CEIL_DIV(mat.m_sizes.m_w, BLOCK_SIZE * 2));

    ReduceUnrollLastWarpKernel<BLOCK_SIZE><<<global_size, block_size>>>(mat.GetMatAllOnCUDAMem(), sum_mat_gpu.GetMatAllOnCUDAMem());

    Mat sum_mat_cpu = sum_mat_gpu.clone(MemType::MEM_CPU);

    sum = 0;
    for (int32_t i = 0; i < global_size.x; ++i)
    {
        sum += sum_mat_cpu.at<int32_t>(0, i);
    }

    return RET_OK;
}
} // namespace cudaup