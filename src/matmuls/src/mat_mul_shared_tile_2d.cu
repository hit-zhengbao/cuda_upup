#include "mat_mul.h"

// 使用shared memory: 使用了共享内存, 使用2维的tile
// 参考实现 https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/11_gemm_optimize/01_tiled2d/README.md

namespace cudaup
{
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int BLOCK_M_TILE, const int BLOCK_N_TILE>
__global__ void MatMulSharedTile2DKernel(Mat *mat0, Mat *mat1, Mat *dst)
{
    const int32_t dst_y = blockIdx.y * BLOCK_M;
    const int32_t dst_x = blockIdx.x * BLOCK_N;

    if ((dst_x + BLOCK_N) > dst->m_sizes.m_w || (dst_y + BLOCK_M) > dst->m_sizes.m_h)
    {
        return;
    }

    // Index of sub-matrix
    const int32_t mat0_thread_y = threadIdx.x / BLOCK_K;    // 0 ~ 7
    const int32_t mat0_thread_x = threadIdx.x % BLOCK_K;    // 0 ~ 7
    const int32_t mat1_thread_y = threadIdx.x / BLOCK_N;    // 0
    const int32_t mat1_thread_x = threadIdx.x % BLOCK_N;    // 0 ~ 63
    const int32_t thread_row    = threadIdx.x / (BLOCK_N / BLOCK_N_TILE);   // 0 ~ 7
    const int32_t thread_col    = threadIdx.x % (BLOCK_N / BLOCK_N_TILE);   // 0 ~ 7

    // Allocate the shared memory for the block
    __shared__ float shared_mat0[BLOCK_M * BLOCK_K];
    __shared__ float shared_mat1[BLOCK_K * BLOCK_N];

    float sum[BLOCK_M_TILE * BLOCK_N_TILE] = {0.f};
    float reg_m[BLOCK_M_TILE]              = {0.f};
    float reg_n[BLOCK_N_TILE]              = {0.f};

    const int32_t number_threads_per_block = blockDim.x;
    const int32_t stride_mat0              = number_threads_per_block / BLOCK_K;    // 8
    const int32_t stride_mat1              = number_threads_per_block / BLOCK_N;    // 1

    for (int32_t i = 0; i < mat0->m_sizes.m_w; i += BLOCK_K)
    {
        // Load the sub-matrix of input matrix into shared memory
        for (int32_t j = 0; j < BLOCK_M; j += stride_mat0)
        {
            shared_mat0[(mat0_thread_y + j) * BLOCK_K + mat0_thread_x] = mat0->at<float>(dst_y + mat0_thread_y + j, i + mat0_thread_x);
        }

        for (int32_t j = 0; j < BLOCK_K; j += stride_mat1)
        {
            shared_mat1[(mat1_thread_y + j) * BLOCK_N + mat1_thread_x] = mat1->at<float>(i + mat1_thread_y + j, dst_x + mat1_thread_x);
        }

        // wait for all threads to finish loading
        __syncthreads();

        // Compute the product of the two matrices of the sub-matrix
        for (int32_t j = 0; j < BLOCK_K; ++j)
        {
            // load the data from shared memory to registers
            for (int32_t k = 0; k < BLOCK_M_TILE; ++k)
            {
                reg_m[k] = shared_mat0[(thread_row * BLOCK_M_TILE + k) * BLOCK_K + j];
            }

            for (int32_t k = 0; k < BLOCK_N_TILE; ++k)
            {
                reg_n[k] = shared_mat1[(j * BLOCK_N + thread_col * BLOCK_N_TILE + k)];
            }

            for (int32_t idx_y = 0; idx_y < BLOCK_M_TILE; ++idx_y)
            {
                for (int32_t idx_x = 0; idx_x < BLOCK_N_TILE; ++idx_x)
                {
                    sum[idx_y * BLOCK_N_TILE + idx_x] += reg_m[idx_y] * reg_n[idx_x];
                }
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (int32_t i = 0; i < BLOCK_M_TILE; ++i)
    {
        for (int32_t j = 0; j < BLOCK_N_TILE; ++j)
        {
            dst->at<float>(dst_y + thread_row * BLOCK_M_TILE + i, dst_x + thread_col * BLOCK_N_TILE + j) = sum[i * BLOCK_N_TILE + j];
        }
    }
}

int32_t MatMulSharedTile2D(Mat &mat0, Mat &mat1, Mat &dst)
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

    const int32_t BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 8, BLOCK_M_TILE = 8, BLOCK_N_TILE = 8;
    dim3          grid_size(CEIL_DIV(w, BLOCK_N), CEIL_DIV(h, BLOCK_M));
    dim3          block_size_kernel(BLOCK_M * BLOCK_N / (BLOCK_M_TILE * BLOCK_N_TILE));

    MatMulSharedTile2DKernel<BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_TILE, BLOCK_N_TILE><<<grid_size, block_size_kernel>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());

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

