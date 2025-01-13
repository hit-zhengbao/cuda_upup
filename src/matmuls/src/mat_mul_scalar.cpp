#include "mat_mul.h"
#include "log.h"
#include "mat.h"

namespace cudaup
{
int32_t MatMulScalar(Mat &mat0, Mat &mat1, Mat &dst)
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
    int32_t mat0_col = mat0.m_sizes.m_w;

    for (int32_t y = 0; y < h; ++y)
    {
        for (int32_t x = 0; x < w; ++x)
        {
            float sum = 0.f;

            for (int32_t k = 0; k < mat0_col; ++k)
            {
                sum += mat0.at<float>(y, k) * mat1.at<float>(k, x);
            }

            dst.at<float>(y, x) = sum;
        }
    }

    ret = RET_OK;
    return ret;
}
}