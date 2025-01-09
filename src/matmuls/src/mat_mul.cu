#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

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

// template<int BLOCK_SIZE>
// __global__ void MatMulSharedMem(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num)
// {
//     __shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float B[BLOCK_SIZE][BLOCK_SIZE];

// }

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.x;
    const uint c_col = blockIdx.y;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions
    A += c_row * BLOCKSIZE * K;
    B += c_col * BLOCKSIZE;
    C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;

    float tmp = 0.0f;
    for (int i = 0; i < K; i += BLOCKSIZE)
    {
        // load the next block of the input matrices into shared memory
        A_shared[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];

        // wait for all threads to finish loading
        __syncthreads();

        // compute the partial sum
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_col];
        }

        // wait for all threads to finish computing
        __syncthreads();

        // advance the pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    C[thread_row * N + thread_col] = tmp;
}

void run_sgemm_shared_memory(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

__global__ void MatMulTopLeft(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num)
{
    // 4*4 
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    // int global_x = threadIdx.x;
    // int global_y = threadIdx.y;

    int x = global_x << 2;
    int y = global_y << 2;

    if (x + 4 > w || y + 4 > h)
    {
        return;
    }

    float4 v_zero = MakeFloat4(0);
    float4 v_sum_row0 = v_zero;
    float4 v_sum_row1 = v_zero;
    float4 v_sum_row2 = v_zero;
    float4 v_sum_row3 = v_zero;

    int i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0 + (y + 0) * stride_num + i);
        float4 v_mat0_row1 = *(float4 *)(mat0 + (y + 1) * stride_num + i);
        float4 v_mat0_row2 = *(float4 *)(mat0 + (y + 2) * stride_num + i);
        float4 v_mat0_row3 = *(float4 *)(mat0 + (y + 3) * stride_num + i);

        float4 v_mat1_row0  = *(float4 *)(mat1 + (i + 0) * stride_num + x);
        float4 v_mat1_row1  = *(float4 *)(mat1 + (i + 1) * stride_num + x);
        float4 v_mat1_row2  = *(float4 *)(mat1 + (i + 2) * stride_num + x);
        float4 v_mat1_row3  = *(float4 *)(mat1 + (i + 3) * stride_num + x);

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
        float val_mat0_row0 = *(float *)(mat0 + (y + 0) * stride_num + i);
        float val_mat0_row1 = *(float *)(mat0 + (y + 1) * stride_num + i);
        float val_mat0_row2 = *(float *)(mat0 + (y + 2) * stride_num + i);
        float val_mat0_row3 = *(float *)(mat0 + (y + 3) * stride_num + i);

        float4 v_mat1_row0  = *(float4 *)(mat1 + (i + 0) * stride_num + x);

        v_sum_row0 += MakeFloat4(val_mat0_row0) * v_mat1_row0;
        v_sum_row1 += MakeFloat4(val_mat0_row1) * v_mat1_row0;
        v_sum_row2 += MakeFloat4(val_mat0_row2) * v_mat1_row0;
        v_sum_row3 += MakeFloat4(val_mat0_row3) * v_mat1_row0;
    }


    *(float4 *)(dst + (y + 0) * stride_num + x) = v_sum_row0;
    *(float4 *)(dst + (y + 1) * stride_num + x) = v_sum_row1;
    *(float4 *)(dst + (y + 2) * stride_num + x) = v_sum_row2;
    *(float4 *)(dst + (y + 3) * stride_num + x) = v_sum_row3;

// printf("****x: %2d, y:%2d, mat0_cold: %d, w: %d, h: %d\n", x, y, mat0_col, w, h);
// if (0 == x && 0 == y)
// {
//     printf("val: %f, %f, %f, %f\n", 
//             mat0[0], mat0[1],
//             mat1[0], mat1[1]
//             );

//     printf("dst: %f, %f, %f, %f\n", 
//             dst[0], dst[1],
//             dst[2], dst[3]
//             );
// }
}


// top-right
__global__ void MatMulTopRight(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num, int offset_x)
{
    // rows: 4, cols: 1
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // int global_x = threadIdx.x;
    // int global_y = threadIdx.y;

    int x = global_x + offset_x;
    int y = global_y << 2;

    if (x >= w || y + 4 > h)
    {
        return;
    }

    float val_sum_row0 = 0;
    float val_sum_row1 = 0;
    float val_sum_row2 = 0;
    float val_sum_row3 = 0;

    int i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0 + (y + 0) * stride_num + i);
        float4 v_mat0_row1 = *(float4 *)(mat0 + (y + 1) * stride_num + i);
        float4 v_mat0_row2 = *(float4 *)(mat0 + (y + 2) * stride_num + i);
        float4 v_mat0_row3 = *(float4 *)(mat0 + (y + 3) * stride_num + i);

        float val_mat1_row0  = *(float *)(mat1 + (i + 0) * stride_num + x);
        float val_mat1_row1  = *(float *)(mat1 + (i + 1) * stride_num + x);
        float val_mat1_row2  = *(float *)(mat1 + (i + 2) * stride_num + x);
        float val_mat1_row3  = *(float *)(mat1 + (i + 3) * stride_num + x);

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
        float val_mat0_row0 = *(float *)(mat0 + (y + 0) * stride_num + i);
        float val_mat0_row1 = *(float *)(mat0 + (y + 1) * stride_num + i);
        float val_mat0_row2 = *(float *)(mat0 + (y + 2) * stride_num + i);
        float val_mat0_row3 = *(float *)(mat0 + (y + 3) * stride_num + i);

        float val_mat1_row0  = *(float *)(mat1 + (i + 0) * stride_num + x);

        val_sum_row0 += val_mat0_row0 * val_mat1_row0;
        val_sum_row1 += val_mat0_row1 * val_mat1_row0;
        val_sum_row2 += val_mat0_row2 * val_mat1_row0;
        val_sum_row3 += val_mat0_row3 * val_mat1_row0;
    }

    *(float *)(dst + (y + 0) * stride_num + x) = val_sum_row0;
    *(float *)(dst + (y + 1) * stride_num + x) = val_sum_row1;
    *(float *)(dst + (y + 2) * stride_num + x) = val_sum_row2;
    *(float *)(dst + (y + 3) * stride_num + x) = val_sum_row3;
}

// down-left
__global__ void MatMulDownLeft(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num, int offset_y)
{
    // rows: 1, cols: 4
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = global_x << 2;
    int y = global_y + offset_y;

    if (x + 4 > w || y >= h)
    {
        return;
    }

    float4 v_sum0 = MakeFloat4(0);
    float4 v_sum1 = MakeFloat4(0);
    float4 v_sum2 = MakeFloat4(0);
    float4 v_sum3 = MakeFloat4(0);

    int i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0 + y * stride_num + i);

        float4 v_mat1_row0  = *(float4 *)(mat1 + (i + 0) * stride_num + x);
        float4 v_mat1_row1  = *(float4 *)(mat1 + (i + 1) * stride_num + x);
        float4 v_mat1_row2  = *(float4 *)(mat1 + (i + 2) * stride_num + x);
        float4 v_mat1_row3  = *(float4 *)(mat1 + (i + 3) * stride_num + x);

        v_sum0 += MakeFloat4(v_mat0_row0.x) * v_mat1_row0;
        v_sum1 += MakeFloat4(v_mat0_row0.y) * v_mat1_row1;
        v_sum2 += MakeFloat4(v_mat0_row0.z) * v_mat1_row2;
        v_sum3 += MakeFloat4(v_mat0_row0.w) * v_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0 + y * stride_num + i);

        float4 v_mat1_row0  = *(float4 *)(mat1 + i * stride_num + x);

        v_sum0 += MakeFloat4(val_mat0_row0) * v_mat1_row0;
    }

    v_sum0 += v_sum1;
    v_sum2 += v_sum3;

    v_sum0 += v_sum2;

    *(float4 *)(dst + y * stride_num + x) = v_sum0;
}

// down-right
__global__ void MatMulDownRight(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num, int offset_y, int offset_x)
{
    // rows: 1, cols: 1
    // int global_x = threadIdx.x;
    // int global_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = global_x + offset_x;
    int y = global_y + offset_y;

    if (x >= w || y >= h)
    {
        return;
    }

    float val_sum0 = 0;
    float val_sum1 = 0;
    float val_sum2 = 0;
    float val_sum3 = 0;

    int i = 0;
    for (; i + 4 <= mat0_col; i += 4)
    {
        float4 v_mat0_row0 = *(float4 *)(mat0 + y * stride_num + i);

        float val_mat1_row0  = *(float *)(mat1 + (i + 0) * stride_num + x);
        float val_mat1_row1  = *(float *)(mat1 + (i + 1) * stride_num + x);
        float val_mat1_row2  = *(float *)(mat1 + (i + 2) * stride_num + x);
        float val_mat1_row3  = *(float *)(mat1 + (i + 3) * stride_num + x);

        val_sum0 += v_mat0_row0.x * val_mat1_row0;
        val_sum1 += v_mat0_row0.y * val_mat1_row1;
        val_sum2 += v_mat0_row0.z * val_mat1_row2;
        val_sum3 += v_mat0_row0.w * val_mat1_row3;
    }

    for (; i < mat0_col; ++i)
    {
        float val_mat0_row0 = *(float *)(mat0 + y * stride_num + i);

        float val_mat1_row0  = *(float *)(mat1 + i * stride_num + x);

        val_sum0 += val_mat0_row0 * val_mat1_row0;
    }

    val_sum0 += val_sum1;
    val_sum2 += val_sum3;

    val_sum0 += val_sum2;

    *(float *)(dst + y * stride_num + x) = val_sum0;
}

void MatMulScalar(float *mat0, float *mat1, float *dst, int mat0_col, int w, int h, int stride_num)
{
    if (!mat0 || !mat1)
    {
        std::cout << "nullptr " << std::endl;
        return;
    }

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float sum = 0.f;

            for (int k = 0; k < mat0_col; ++k)
            {
                sum += mat0[y * stride_num + k] * mat1[k * stride_num + x];
            }

            dst[y * stride_num + x] = sum;
        }
    }
}

void CheckMatVal(float *mat0, float *mat1, int w, int h, int stride_num)
{
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float diff = std::abs(mat0[y * stride_num + x] - mat1[y * stride_num + x]);
            if (diff > 1e-6)
            {
                std::cout << "mat0: " << mat0[y * stride_num + x] << " ,mat1: " << mat1[y * stride_num + x] << ", diff: " << diff \
                 << " , y: " << y  << " , x: " << x << std::endl;
                return;
            }
        }
    }

    std::cout << "**** Cmp Ok****" << std::endl;
}

// int main(int argc, char **argv)
// {
//     if (argc < 3)
//     {
//         printf("Usage: ./matmul stride_num size\n");
//         return -1;
//     }

//     // int w = size, h = size, mat0_col = size;
//     int w = std::stoi(argv[2]), h = w, mat0_col = w, stride_num = std::stoi(argv[1]), size = stride_num;
//     printf("*** w: %d, h: %d, stride_num: %d\n", w, h, stride_num);

//     std::vector<float> mat0(size * size);
//     std::vector<float> mat1(size * size);

//     std::vector<float> dst_c(size * size);
//     std::vector<float> dst_host_cu(size * size);

//     std::srand(std::time(nullptr));

//     for (size_t i = 0; i < mat0.size(); ++i)
//     {
//         // mat0[i] = std::rand() % size;
//         // mat1[i] = std::rand() % size;
//         mat0[i] = i % 30;
//         mat1[i] = i % 30;
//     }

//     // gold
//     MatMulScalar(mat0.data(), mat1.data(), dst_c.data(), mat0_col, w, h, stride_num);

//     // cuda
//     // Allocate CUDA mem and copy
//     float *mat0_cu, *mat1_cu, *dst_cu;
//     size_t bytes_size = mat0.size() * sizeof(float);

//     cudaMalloc((void **)&mat0_cu,   bytes_size);
//     cudaMalloc((void **)&mat1_cu,   bytes_size);
//     cudaMalloc((void **)&dst_cu,    bytes_size);

//     cudaMemcpy(mat0_cu, mat0.data(), bytes_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(mat1_cu, mat1.data(), bytes_size, cudaMemcpyHostToDevice);

//     int div_x = w >> 2, div_y = h >> 2;
//     int offset_x = div_x << 2;
//     int offset_y = div_y << 2;

//     int left_x   = w - offset_x;
//     int left_y   = h - offset_y;

//     dim3 block_size(4, 4); 
//     dim3 grid_size((div_x + block_size.x - 1) / block_size.x, (div_y + block_size.y - 1) / block_size.y);
    
//     printf("*** block size: %d*%d, grid size: %d*%d\n", block_size.x, block_size.y, grid_size.x, grid_size.y);

//     // top left
//     MatMulTopLeft<<<grid_size, block_size>>>(mat0_cu, mat1_cu, dst_cu, mat0_col, w, h, stride_num);

//     // top right
//     if (offset_x < w)
//     {
//         dim3 block_tr_size(4, 4);
//         dim3 grid_tr_size((left_x + block_tr_size.x - 1) / block_tr_size.x, (div_y + block_tr_size.y - 1) / block_tr_size.y);
//         MatMulTopRight<<<grid_tr_size, block_tr_size>>>(mat0_cu, mat1_cu, dst_cu, mat0_col, w, h, stride_num, offset_x);
//     }

//     // down left
//     if (offset_y < h)
//     {
//         dim3 block_dl_size(4, 4);
//         dim3 grid_dl_size((div_x + block_dl_size.x - 1) / block_dl_size.x, (left_y + block_dl_size.y - 1) / block_dl_size.y);
//         MatMulDownLeft<<<grid_dl_size, block_dl_size>>>(mat0_cu, mat1_cu, dst_cu, mat0_col, w, h, stride_num, offset_y);
//     }

//     // down right
//     if (offset_x < w && offset_y < h)
//     {
//         dim3 block_dr_size(4, 4);
//         dim3 grid_dr_size((left_x + block_dr_size.x - 1) / block_dr_size.x, (left_y + block_dr_size.y - 1) / block_dr_size.y);
//         MatMulDownRight<<<grid_dr_size, block_dr_size>>>(mat0_cu, mat1_cu, dst_cu, mat0_col, w, h, stride_num, offset_y, offset_x);
//     }

//     // Sync CUDA
//     cudaError_t err = cudaDeviceSynchronize();
//     if (err != cudaSuccess)
//     {
//         printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
//     }

//     cudaMemcpy(dst_host_cu.data(), dst_cu, bytes_size, cudaMemcpyDeviceToHost);

//     // Compare
//     CheckMatVal(dst_c.data(), dst_host_cu.data(), w, h, stride_num);

//     {
//         run_sgemm_shared_memory(mat0_cu, mat1_cu, dst_cu, h, w, mat0_col);

//         // Sync CUDA
//         cudaError_t err = cudaDeviceSynchronize();
//         if (err != cudaSuccess)
//         {
//             printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
//         }
//     }

//     cudaFree(mat0_cu);
//     cudaFree(mat1_cu);
//     cudaFree(dst_cu);

//     return 0;
// }
