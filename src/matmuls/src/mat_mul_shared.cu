#include "mat_mul.h"

// 使用shared memory: 使用了共享内存, 但每个线程计算一个点
// 参考实现https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/07_optimize_matmul/matmul_shared.cu

namespace cudaup
{
template<const int BLOCK_SIZE>
__global__ void MatMulSharedKernel(Mat *mat0, Mat *mat1, Mat *dst)
{
    const int32_t dst_y = blockIdx.y * BLOCK_SIZE;
    const int32_t dst_x = blockIdx.x * BLOCK_SIZE;

    if ((dst_x + BLOCK_SIZE) > dst->m_sizes.m_w || (dst_y + BLOCK_SIZE) > dst->m_sizes.m_h)
    {
        return;
    }

    // Index of sub-matrix
    const int32_t thread_y  = threadIdx.x / BLOCK_SIZE;
    const int32_t thread_x  = threadIdx.x % BLOCK_SIZE;

    // Allocate the shared memory for the block
    __shared__ float shared_mat0[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_mat1[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.f;

    for (int32_t i = 0; i < mat0->m_sizes.m_w; i += BLOCK_SIZE)
    {
        // Load the sub-matrix of input matrix into shared memory
        shared_mat0[thread_y * BLOCK_SIZE + thread_x] = mat0->at<float>(dst_y + thread_y, i + thread_x);
        shared_mat1[thread_y * BLOCK_SIZE + thread_x] = mat1->at<float>(i + thread_y, dst_x + thread_x);

        // wait for all threads to finish loading
        __syncthreads();

        // Compute the product of the two matrices of the sub-matrix
        for (int32_t j = 0; j < BLOCK_SIZE; ++j)
        {
            sum += shared_mat0[thread_y * BLOCK_SIZE + j] * shared_mat1[j * BLOCK_SIZE + thread_x];
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    dst->at<float>(dst_y + thread_y, dst_x + thread_x) = sum;
}

int32_t MatMulShared(Mat &mat0, Mat &mat1, Mat &dst)
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

    const int32_t block_size = 32;
    dim3          grid_size((h + block_size - 1) / block_size, (w + block_size - 1) / block_size);
    dim3          block_size_kernel(block_size * block_size);

    MatMulSharedKernel<block_size><<<grid_size, block_size_kernel>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());

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

