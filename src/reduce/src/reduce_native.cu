#include "reduce.h"

namespace cudaup
{
template<int32_t BLOCK_SIZE>
CUDA_GLOBAL void ReduceNativeKernel(Mat *mat, int32_t *sum)
{
    __shared__ int32_t shared_data[BLOCK_SIZE];

    int32_t t_id  = threadIdx.x;
    int32_t b_id  = blockIdx.x;
    int32_t b_dim = blockDim.x;


}

int32_t ReduceNative(Mat &mat, int32_t &sum)
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


    return RET_OK;
}
} // namespace cudaup