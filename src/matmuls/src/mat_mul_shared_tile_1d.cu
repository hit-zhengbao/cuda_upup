#include "mat_mul.h"

// 使用shared memory: 使用了共享内存,
// 参考实现

namespace cudaup
{
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int BLOCK_K_TILE>
__global__ void MatMulSharedTile1DKernel(Mat *mat0, Mat *mat1, Mat *dst)
{
    const int32_t dst_y = blockIdx.y * BLOCK_M;
    const int32_t dst_x = blockIdx.x * BLOCK_N;

    // if ((dst_x + BLOCK_SIZE) > dst->m_sizes.m_w || (dst_y + BLOCK_SIZE/4) > dst->m_sizes.m_h)
    // {
    //     return;
    // }

    // Index of sub-matrix
    const int32_t mat0_thread_y = threadIdx.x / BLOCK_K;
    const int32_t mat0_thread_x = threadIdx.x % BLOCK_K;
    const int32_t mat1_thread_y = threadIdx.x / BLOCK_N;
    const int32_t mat1_thread_x = threadIdx.x % BLOCK_N;

    // Allocate the shared memory for the block
    __shared__ float shared_mat0[BLOCK_M * BLOCK_K];
    __shared__ float shared_mat1[BLOCK_K * BLOCK_N];

    float sum[BLOCK_K_TILE] = {0.f};

    for (int32_t i = 0; i < mat0->m_sizes.m_w; i += BLOCK_K)
    {
        // Load the sub-matrix of input matrix into shared memory
        shared_mat0[mat0_thread_y * BLOCK_K + mat0_thread_x] = mat0->at<float>(dst_y + mat0_thread_y, i + mat0_thread_x);
        shared_mat1[mat1_thread_y * BLOCK_N + mat1_thread_x] = mat1->at<float>(i + mat1_thread_y, dst_x + mat1_thread_x);

        // wait for all threads to finish loading
        __syncthreads();

        // Compute the product of the two matrices of the sub-matrix
        for (int32_t j = 0; j < BLOCK_K; ++j)
        {
            // sum += shared_mat0[thread_y * BLOCK_SIZE + j] * shared_mat1[j * BLOCK_SIZE + thread_x];
            float tmp_val_mat1 = shared_mat1[j * BLOCK_N + mat1_thread_x];

            for (int32_t k = 0; k < BLOCK_K_TILE; ++k)
            {
                sum[k] += shared_mat0[(mat0_thread_y + k) * BLOCK_K + j] * tmp_val_mat1;
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (int32_t i = 0; i < BLOCK_K_TILE; ++i)
    {
        dst->at<float>(dst_y + mat0_thread_y + i, dst_x + mat1_thread_x) = sum[i];
    }
}

int32_t MatMulSharedTile1D(Mat &mat0, Mat &mat1, Mat &dst)
{
    int32_t ret = RET_ERR;

    if (mat0.empty() || mat1.empty() || dst.empty())
    {
        LOG_ERROR("MatMulScalar: mat0 or mat1 or dst is null!");
        return ret;
    }

    // only support fp32c1
    if (mat0.m_sizes.m_ch != 1 || mat1.m_sizes.m_ch != 1 || dst.m_sizes.m_ch != 1)
    {
        LOG_ERROR("MatMulScalar: only support fp32c1!");
        return ret;
    }

    // Only support w = 32*x && h = 32*x
    if ((mat0.m_sizes.m_w & 31) != 0 || (mat0.m_sizes.m_h & 31) != 0 || (mat1.m_sizes.m_w & 31) != 0 || (mat1.m_sizes.m_h & 31) != 0
        || (dst.m_sizes.m_w & 31) != 0 || (dst.m_sizes.m_h & 31) != 0)
    {
        LOG_ERROR("MatMulScalar: only support w = 32*x && h = 32*x!");
        return ret;
    }

    int32_t h = mat0.m_sizes.m_h;
    int32_t w = mat1.m_sizes.m_w;

    const int32_t BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 8, BLOCK_K_TILE = 8;
    dim3          grid_size(CEIL_DIV(w, BLOCK_N), CEIL_DIV(h, BLOCK_M));
    dim3          block_size_kernel(BLOCK_M * BLOCK_N / BLOCK_K_TILE);

    MatMulSharedTile1DKernel<block_size><<<grid_size, block_size_kernel>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());

    // Sync CUDA
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA kernel error: %s", cudaGetErrorString(err));
        return ret;
    }

    ret = RET_OK;
    return ret;
}

}

