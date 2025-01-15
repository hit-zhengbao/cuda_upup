#include "mat_mul.h"

namespace cudaup
{

/** 
 * 矩阵乘法，使用block和no local memory
*/
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__device__ float4 MakeFloat4(float val)
{
    return make_float4(val, val, val, val);
}

__device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ float4& operator+=(float4 &a, float4 b)
{
    a = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    return a;
}

// Result: 4 * 4
__global__ void MatMulTopLeft(Mat *mat0, Mat *mat1, Mat *dst)
{
    // 4*4 
    int32_t global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t x = global_x << 2;
    int32_t y = global_y << 2;

    int32_t h        = mat0->m_sizes.m_h;
    int32_t w        = mat1->m_sizes.m_w;
    int32_t mat0_col = mat0->m_sizes.m_w;

    if (x + 4 > w || y + 4 > h)
    {
        return;
    }

    float4 v_zero = MakeFloat4(0.f);
    float4 v_sum_row0 = v_zero;
    float4 v_sum_row1 = v_zero;
    float4 v_sum_row2 = v_zero;
    float4 v_sum_row3 = v_zero;

    int32_t i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0->ptr<float>(y + 0, i));
        float4 v_mat0_row1 = *(float4 *)(mat0->ptr<float>(y + 1, i));
        float4 v_mat0_row2 = *(float4 *)(mat0->ptr<float>(y + 2, i));
        float4 v_mat0_row3 = *(float4 *)(mat0->ptr<float>(y + 3, i));

        float4 v_mat1_row0  = *(float4 *)(mat1->ptr<float>((i + 0), x));
        float4 v_mat1_row1  = *(float4 *)(mat1->ptr<float>((i + 1), x));
        float4 v_mat1_row2  = *(float4 *)(mat1->ptr<float>((i + 2), x));
        float4 v_mat1_row3  = *(float4 *)(mat1->ptr<float>((i + 3), x));

        v_sum_row0 += MakeFloat4(v_mat0_row0.x) * v_mat1_row0;
        v_sum_row1 += MakeFloat4(v_mat0_row1.x) * v_mat1_row0;
        v_sum_row2 += MakeFloat4(v_mat0_row2.x) * v_mat1_row0;
        v_sum_row3 += MakeFloat4(v_mat0_row3.x) * v_mat1_row0;

        v_sum_row0 += MakeFloat4(v_mat0_row0.y) * v_mat1_row1;
        v_sum_row1 += MakeFloat4(v_mat0_row1.y) * v_mat1_row1;
        v_sum_row2 += MakeFloat4(v_mat0_row2.y) * v_mat1_row1;
        v_sum_row3 += MakeFloat4(v_mat0_row3.y) * v_mat1_row1;

        v_sum_row0 += MakeFloat4(v_mat0_row0.z) * v_mat1_row2;
        v_sum_row1 += MakeFloat4(v_mat0_row1.z) * v_mat1_row2;
        v_sum_row2 += MakeFloat4(v_mat0_row2.z) * v_mat1_row2;
        v_sum_row3 += MakeFloat4(v_mat0_row3.z) * v_mat1_row2;

        v_sum_row0 += MakeFloat4(v_mat0_row0.w) * v_mat1_row3;
        v_sum_row1 += MakeFloat4(v_mat0_row1.w) * v_mat1_row3;
        v_sum_row2 += MakeFloat4(v_mat0_row2.w) * v_mat1_row3;
        v_sum_row3 += MakeFloat4(v_mat0_row3.w) * v_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0->ptr<float>(y + 0, i));
        float val_mat0_row1 = *(float *)(mat0->ptr<float>(y + 1, i));
        float val_mat0_row2 = *(float *)(mat0->ptr<float>(y + 2, i));
        float val_mat0_row3 = *(float *)(mat0->ptr<float>(y + 3, i));

        float4 v_mat1_row0  = *(float4 *)(mat1->ptr<float>(i + 0, x));

        v_sum_row0 += MakeFloat4(val_mat0_row0) * v_mat1_row0;
        v_sum_row1 += MakeFloat4(val_mat0_row1) * v_mat1_row0;
        v_sum_row2 += MakeFloat4(val_mat0_row2) * v_mat1_row0;
        v_sum_row3 += MakeFloat4(val_mat0_row3) * v_mat1_row0;
    }


    *(float4 *)(dst->ptr<float>(y + 0, x)) = v_sum_row0;
    *(float4 *)(dst->ptr<float>(y + 1, x)) = v_sum_row1;
    *(float4 *)(dst->ptr<float>(y + 2, x)) = v_sum_row2;
    *(float4 *)(dst->ptr<float>(y + 3, x)) = v_sum_row3;
}

// top-right
// Result: 4 * 1
__global__ void MatMulTopRight(Mat *mat0, Mat *mat1, Mat *dst)
{
    int32_t global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t h        = mat0->m_sizes.m_h;
    int32_t w        = mat1->m_sizes.m_w;
    int32_t mat0_col = mat0->m_sizes.m_w;
    int32_t offset_x = (w >> 2) << 2;

    int32_t x = global_x + offset_x;
    int32_t y = global_y << 2;

    if (x >= w || y + 4 > h)
    {
        return;
    }

    float val_sum_row0 = 0;
    float val_sum_row1 = 0;
    float val_sum_row2 = 0;
    float val_sum_row3 = 0;

    int32_t i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0->ptr<float>(y + 0, i));
        float4 v_mat0_row1 = *(float4 *)(mat0->ptr<float>(y + 1, i));
        float4 v_mat0_row2 = *(float4 *)(mat0->ptr<float>(y + 2, i));
        float4 v_mat0_row3 = *(float4 *)(mat0->ptr<float>(y + 3, i));

        float val_mat1_row0  = *(float *)(mat1->ptr<float>(i + 0, x));
        float val_mat1_row1  = *(float *)(mat1->ptr<float>(i + 1, x));
        float val_mat1_row2  = *(float *)(mat1->ptr<float>(i + 2, x));
        float val_mat1_row3  = *(float *)(mat1->ptr<float>(i + 3, x));

        val_sum_row0 += v_mat0_row0.x * val_mat1_row0;
        val_sum_row1 += v_mat0_row1.x * val_mat1_row0;
        val_sum_row2 += v_mat0_row2.x * val_mat1_row0;
        val_sum_row3 += v_mat0_row3.x * val_mat1_row0;

        val_sum_row0 += v_mat0_row0.y * val_mat1_row1;
        val_sum_row1 += v_mat0_row1.y * val_mat1_row1;
        val_sum_row2 += v_mat0_row2.y * val_mat1_row1;
        val_sum_row3 += v_mat0_row3.y * val_mat1_row1;

        val_sum_row0 += v_mat0_row0.z * val_mat1_row2;
        val_sum_row1 += v_mat0_row1.z * val_mat1_row2;
        val_sum_row2 += v_mat0_row2.z * val_mat1_row2;
        val_sum_row3 += v_mat0_row3.z * val_mat1_row2;

        val_sum_row0 += v_mat0_row0.w * val_mat1_row3;
        val_sum_row1 += v_mat0_row1.w * val_mat1_row3;
        val_sum_row2 += v_mat0_row2.w * val_mat1_row3;
        val_sum_row3 += v_mat0_row3.w * val_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0->ptr<float>(y + 0, i));
        float val_mat0_row1 = *(float *)(mat0->ptr<float>(y + 1, i));
        float val_mat0_row2 = *(float *)(mat0->ptr<float>(y + 2, i));
        float val_mat0_row3 = *(float *)(mat0->ptr<float>(y + 3, i));

        float val_mat1_row0  = *(float *)(mat1->ptr<float>(i + 0, x));

        val_sum_row0 += val_mat0_row0 * val_mat1_row0;
        val_sum_row1 += val_mat0_row1 * val_mat1_row0;
        val_sum_row2 += val_mat0_row2 * val_mat1_row0;
        val_sum_row3 += val_mat0_row3 * val_mat1_row0;
    }

    *(float *)(dst->ptr<float>(y + 0, x)) = val_sum_row0;
    *(float *)(dst->ptr<float>(y + 1, x)) = val_sum_row1;
    *(float *)(dst->ptr<float>(y + 2, x)) = val_sum_row2;
    *(float *)(dst->ptr<float>(y + 3, x)) = val_sum_row3;
}

// down-left
// Result: 1 * 4
__global__ void MatMulDownLeft(Mat *mat0, Mat *mat1, Mat *dst)
{
    int32_t global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t h        = mat0->m_sizes.m_h;
    int32_t w        = mat1->m_sizes.m_w;
    int32_t mat0_col = mat0->m_sizes.m_w;
    int32_t offset_y = (h >> 2) << 2;

    int32_t x = global_x << 2;
    int32_t y = global_y + offset_y;

    if (x + 4 > w || y >= h)
    {
        return;
    }

    float4 v_sum0 = MakeFloat4(0);
    float4 v_sum1 = MakeFloat4(0);
    float4 v_sum2 = MakeFloat4(0);
    float4 v_sum3 = MakeFloat4(0);

    int32_t i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0->ptr<float>(y, i));

        float4 v_mat1_row0  = *(float4 *)(mat1->ptr<float>(i + 0, x));
        float4 v_mat1_row1  = *(float4 *)(mat1->ptr<float>(i + 1, x));
        float4 v_mat1_row2  = *(float4 *)(mat1->ptr<float>(i + 2, x));
        float4 v_mat1_row3  = *(float4 *)(mat1->ptr<float>(i + 3, x));

        v_sum0 += MakeFloat4(v_mat0_row0.x) * v_mat1_row0;
        v_sum1 += MakeFloat4(v_mat0_row0.y) * v_mat1_row1;
        v_sum2 += MakeFloat4(v_mat0_row0.z) * v_mat1_row2;
        v_sum3 += MakeFloat4(v_mat0_row0.w) * v_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0->ptr<float>(y, i));

        float4 v_mat1_row0  = *(float4 *)(mat1->ptr<float>(i, x));

        v_sum0 += MakeFloat4(val_mat0_row0) * v_mat1_row0;
    }

    v_sum0 += v_sum1;
    v_sum2 += v_sum3;

    v_sum0 += v_sum2;

    *(float4 *)(dst->ptr<float>(y, x)) = v_sum0;
}

// down-right
// Result: 1 * 1
__global__ void MatMulDownRight(Mat *mat0, Mat *mat1, Mat *dst)
{
    int32_t global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t h        = mat0->m_sizes.m_h;
    int32_t w        = mat1->m_sizes.m_w;
    int32_t mat0_col = mat0->m_sizes.m_w;
    int32_t offset_y = (h >> 2) << 2;
    int32_t offset_x = (w >> 2) << 2;

    int32_t x = global_x + offset_x;
    int32_t y = global_y + offset_y;

    if (x >= w || y >= h)
    {
        return;
    }

    float val_sum0 = 0;
    float val_sum1 = 0;
    float val_sum2 = 0;
    float val_sum3 = 0;

    int32_t i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0->ptr<float>(y, i));

        float val_mat1_row0  = *(float *)(mat1->ptr<float>(i + 0, x));
        float val_mat1_row1  = *(float *)(mat1->ptr<float>(i + 1, x));
        float val_mat1_row2  = *(float *)(mat1->ptr<float>(i + 2, x));
        float val_mat1_row3  = *(float *)(mat1->ptr<float>(i + 3, x));

        val_sum0 += v_mat0_row0.x * val_mat1_row0;
        val_sum1 += v_mat0_row0.y * val_mat1_row1;
        val_sum2 += v_mat0_row0.z * val_mat1_row2;
        val_sum3 += v_mat0_row0.w * val_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0->ptr<float>(y, i));

        float val_mat1_row0  = *(float *)(mat1->ptr<float>(i, x));

        val_sum0 += val_mat0_row0 * val_mat1_row0;
    }

    val_sum0 += val_sum1;
    val_sum2 += val_sum3;

    val_sum0 += val_sum2;

    *(float *)(dst->ptr<float>(y, x)) = val_sum0;
}

int32_t MatMulBlockAndNoLocalMem(Mat &mat0, Mat &mat1, Mat &dst)
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

    int32_t h        = mat0.m_sizes.m_h;
    int32_t w        = mat1.m_sizes.m_w;

    int32_t div_x = w >> 2, div_y = h >> 2;
    int32_t offset_x = div_x << 2;
    int32_t offset_y = div_y << 2;

    int32_t left_x   = w - offset_x;
    int32_t left_y   = h - offset_y;

    dim3 block_size(4, 4); 
    dim3 grid_size((div_x + block_size.x - 1) / block_size.x, (div_y + block_size.y - 1) / block_size.y);
    
    LOG_DEBUG("*** block size: %d*%d, grid size: %d*%d", block_size.x, block_size.y, grid_size.x, grid_size.y);

    // top left
    MatMulTopLeft<<<grid_size, block_size>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());

    // top right
    if (offset_x < w)
    {
        dim3 block_tr_size(4, 4);
        dim3 grid_tr_size((left_x + block_tr_size.x - 1) / block_tr_size.x, (div_y + block_tr_size.y - 1) / block_tr_size.y);
        MatMulTopRight<<<grid_tr_size, block_tr_size>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());
    }

    // down left
    if (offset_y < h)
    {
        dim3 block_dl_size(4, 4);
        dim3 grid_dl_size((div_x + block_dl_size.x - 1) / block_dl_size.x, (left_y + block_dl_size.y - 1) / block_dl_size.y);
        MatMulDownLeft<<<grid_dl_size, block_dl_size>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());
    }

    // down right
    if (offset_x < w && offset_y < h)
    {
        dim3 block_dr_size(4, 4);
        dim3 grid_dr_size((left_x + block_dr_size.x - 1) / block_dr_size.x, (left_y + block_dr_size.y - 1) / block_dr_size.y);
        MatMulDownRight<<<grid_dr_size, block_dr_size>>>(mat0.GetMatAllOnCUDAMem(), mat1.GetMatAllOnCUDAMem(), dst.GetMatAllOnCUDAMem());
    }

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