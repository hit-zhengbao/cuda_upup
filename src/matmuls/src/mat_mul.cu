#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>


/** 
 * 矩阵乘法，使用block和no local memory
*/
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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
