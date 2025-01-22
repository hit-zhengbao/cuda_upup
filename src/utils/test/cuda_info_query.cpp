#include "simple_gtest.h"

#include <cuda_runtime.h>
#include <iostream>


SIMPLE_TEST(cuda_device_info)
{
    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Total max threads: "
                  << prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}
