#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n)
{
    // int index = threadIdx.x;
    // int stride = blockDim.x;

    // for (int i = index; i < n; i += stride)
    // {
    //     out[i] = x[i] + y[i];
    // }

    int id =  blockDim.x * blockIdx.x +  threadIdx.x;

    if (id < n)
    {
        out[id] = x[id] + y[id];
    }
}

int main()
{
    const int num = 1000000;
    size_t bytes_size = num * sizeof(float);

    float *x_buf    = (float *)malloc(bytes_size);
    float *y_buf    = (float *)malloc(bytes_size);
    float *out_buf  = (float *)malloc(bytes_size);

    float *x_cuda_buf, *y_cuda_buf, *out_cuda_buf;

    // initialize 
    for (int i = 0; i < num; ++i)
    {
        x_buf[i] = 1;
        y_buf[i] = 2;
    }

    // Allocate CUDA mem and copy
    cudaMalloc((void **)&x_cuda_buf,            bytes_size);
    cudaMalloc((void **)&y_cuda_buf,            bytes_size);
    cudaMalloc((void **)&out_cuda_buf, bytes_size);

    cudaMemcpy(x_cuda_buf, x_buf, bytes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_cuda_buf, y_buf, bytes_size, cudaMemcpyHostToDevice);

    // Run kernel
    add_kernel<<<1, 256>>>(x_cuda_buf, y_cuda_buf, out_cuda_buf, num);

    // Sync CUDA
    cudaDeviceSynchronize();

    // Copy
    cudaMemcpy(out_buf, out_cuda_buf, bytes_size, cudaMemcpyDeviceToHost);

    // Check result
    for (int i = 0; i < 10; ++i)
    {
        printf("i: %d, result: %.3f\n", i, out_buf[i]);
    }

    cudaFree(x_cuda_buf);
    cudaFree(y_cuda_buf);
    cudaFree(out_cuda_buf);

    free(x_buf);
    free(y_buf);
    free(out_buf);

    return 0;
}
