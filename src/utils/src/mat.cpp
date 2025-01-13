#include "mat.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace cudaup
{
CUDA_HOST_DEVICE Mat::Mat(const Sizes& sizes, MatType type, MemType mem, const Sizes& strides) :
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

CUDA_HOST_DEVICE Mat::Mat(const Mat& other_mat) : m_sizes(other_mat.m_sizes), m_stride(other_mat.m_stride), m_type(other_mat.m_type), m_mem(other_mat.m_mem),
                                                  m_total_bytes(other_mat.m_total_bytes), m_raw_data(other_mat.m_raw_data), m_data(other_mat.m_data),
                                                  m_ref_count(other_mat.m_ref_count)
{
    AddReference(1);
}

CUDA_HOST_DEVICE Mat& Mat::operator=(const Mat& other_mat)
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

CUDA_HOST_DEVICE Mat::~Mat()
{
    Release();
}

CUDA_HOST_DEVICE Mat Mat::clone(MemType mem) const
{
    MemType mem_type = MemType::MEM_INVALID == mem ? m_mem : mem;

    Mat ret_mat(m_sizes, m_type, mem_type, m_stride);

    if (ret_mat.empty())
    {
        LOG_ERROR("Failed to clone mat for no memory");
        return Mat();
    }

    if (MemType::MEM_CPU == m_mem && MemType::MEM_CPU == mem_type)
    {
        memcpy(ret_mat.m_data, m_data, m_total_bytes);
    }
    else if (MemType::MEM_GPU == m_mem && MemType::MEM_GPU == mem_type)
    {
        cudaMemcpy(ret_mat.m_data, m_data, m_total_bytes, cudaMemcpyDeviceToDevice);
    }
    else if (MemType::MEM_GPU == m_mem && MemType::MEM_CPU == mem_type)
    {
        cudaMemcpy(ret_mat.m_data, m_data, m_total_bytes, cudaMemcpyDeviceToHost);
    }
    else if (MemType::MEM_CPU == m_mem && MemType::MEM_GPU == mem_type)
    {
        cudaMemcpy(ret_mat.m_data, m_data, m_total_bytes, cudaMemcpyHostToDevice);
    }

    return ret_mat;
}

CUDA_HOST_DEVICE int32_t Mat::AddReference(int32_t delta)
{
    if (m_ref_count)
    {
        m_ref_count->fetch_add(delta);

        return m_ref_count->load();
    }

    return 0;
}

CUDA_HOST_DEVICE void Mat::Release()
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
