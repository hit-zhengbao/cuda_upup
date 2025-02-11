#include "reduce.h"
#include "log.h"
#include "mat.h"

namespace cudaup
{
// S32C1
int32_t ReduceScalar(Mat &mat, int32_t &sum)
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

    for (int32_t i = 0; i < mat.m_sizes.m_h; ++i)
    {
        for (int32_t j = 0; j < mat.m_sizes.m_w; ++j)
        {
            sum += mat.at<int32_t>(i, j);
        }
    }

    return RET_OK;
}

} // namespace cudaup