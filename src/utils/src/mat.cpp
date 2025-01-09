#include "mat.h"

#include <cuda_runtime.h>

namespace cudaup
{
Mat::Mat(const Sizes& sizes, MatType type, MemType mem, const Sizes& strides) :
         m_sizes(sizes), m_stride(strides), m_type(type), m_mem(mem), m_total_bytes(0),
         m_raw_data(nullptr), m_data(nullptr), m_ref_count(nullptr)
{
    int pitch       = m_sizes.m_w * MatTypeSize(m_type) * m_sizes.m_ch;
    m_stride.m_w    = std::max(pitch, m_stride.m_w);
    m_total_bytes   = pitch * m_sizes.m_h;

    // Init reference count
    m_ref_count = new std::atomic<int32_t>(1);

    if (nullptr == m_ref_count)
    {
        LOG_ERROR("Failed to allocate memory for refer_count");
    }
    else
    {
        // Allocate memory for raw_data
        if (m_mem == MemType::MEM_CPU)
        {
            int32_t aligned_num = 32;

            m_raw_data = new uint8_t[m_total_bytes + aligned_num - 1];

            if (nullptr == m_raw_data)
            {
                LOG_ERROR("Failed to allocate memory for raw_data");
            }
            else
            {
                // Align the data to 32 bytes
                m_data = (uint8_t *)((((size_t)m_raw_data) + aligned_num - 1) & -aligned_num);
            }
        }
        else if (m_mem == MemType::MEM_GPU)
        {
            cudaMalloc(&m_raw_data, m_total_bytes);
            if (nullptr == m_raw_data)
            {
                LOG_ERROR("Failed to allocate memory for raw_data");
            }
            else
            {
                m_data = m_raw_data;
            }
        }
        else
        {
            LOG_ERROR("Unknown memory type");
        }
    }
}

Mat::Mat(const Mat& other_mat) : m_sizes(other_mat.m_sizes), m_stride(other_mat.m_stride), m_type(other_mat.m_type), m_mem(other_mat.m_mem),
                                 m_total_bytes(other_mat.m_total_bytes), m_raw_data(other_mat.m_raw_data), m_data(other_mat.m_data),
                                 m_ref_count(other_mat.m_ref_count)
{
    AddReference(1);
}

Mat& Mat::operator=(const Mat& other_mat)
{
    if (this == &other_mat)
    {
        return *this;
    }

    Release();

    m_sizes       = other_mat.m_sizes;
    m_stride      = other_mat.m_stride;
    m_type        = other_mat.m_type;
    m_mem         = other_mat.m_mem;
    m_total_bytes = other_mat.m_total_bytes;
    m_raw_data    = other_mat.m_raw_data;
    m_data        = other_mat.m_data;
    m_ref_count   = other_mat.m_ref_count;

    AddReference(1);
}

Mat::~Mat()
{
    Release();
}

int32_t Mat::AddReference(int32_t delta)
{
    if (m_ref_count)
    {
        m_ref_count->fetch_add(delta);

        return m_ref_count->load();
    }

    return 0;
}

void Mat::Release()
{
    m_sizes       = 0;
    m_stride      = 0;
    m_type        = MatType::MAT_INVALID;
    m_mem         = MemType::MEM_INVALID;
    m_total_bytes = 0;

    if (m_ref_count != nullptr && AddReference(-1) == 0)
    {
        // Free memory
        if (m_mem == MemType::MEM_CPU)
        {
            delete[] m_raw_data;
        }
        else if (m_mem == MemType::MEM_GPU)
        {
            cudaFree(m_raw_data);
        }

        delete m_ref_count;

        m_raw_data  = nullptr;
        m_data      = nullptr;
        m_ref_count = nullptr;
    }
}

} // namespace cudaup
