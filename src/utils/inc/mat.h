#pragma once

#include "log.h"
#include "def.h"

#include <cstdint>
#include <algorithm>
#include <atomic>
#include <random>

namespace cudaup
{
class Sizes
{
public:
    Sizes() : m_w(0), m_h(0), m_ch(0) {}

    Sizes(int32_t scalar) : m_w(scalar), m_h(scalar), m_ch(scalar) {}

    Sizes(int32_t w, int32_t h) : m_w(w), m_h(h), m_ch(1) {}

    Sizes(int32_t w, int32_t h, int32_t ch) : m_w(w), m_h(h), m_ch(ch) {}

    Sizes(const Sizes& other_sizes) : m_w(other_sizes.m_w), m_h(other_sizes.m_h), m_ch(other_sizes.m_ch) {}

    Sizes& operator=(const Sizes& other_sizes)
    {
        m_w = other_sizes.m_w;
        m_h = other_sizes.m_h;
        m_ch = other_sizes.m_ch;

        return *this;
    }

    CUDA_HOST_DEVICE inline bool equal(const Sizes& other_sizes) const
    {
        return m_w == other_sizes.m_w && m_h == other_sizes.m_h && m_ch == other_sizes.m_ch;
    }

    int32_t m_w;    // width
    int32_t m_h;    // height
    int32_t m_ch;   // channel
};

enum class MatType
{
    MAT_INVALID,
    MAT_U8,
    MAT_S8,
    MAT_U16,
    MAT_S16,
    MAT_U32,
    MAT_S32,
    MAT_F32,
    MAT_F64,
};

enum class MemType
{
    MEM_INVALID,
    MEM_CPU,
    MEM_GPU,
};

static CUDA_HOST_DEVICE int MatTypeSize(MatType type)
{
    switch (type)
    {
        case MatType::MAT_U8:
            return 1;
        case MatType::MAT_S8:
            return 1;
        case MatType::MAT_U16:
            return 2;
        case MatType::MAT_S16:
            return 2;
        case MatType::MAT_U32:
            return 4;
        case MatType::MAT_S32:
            return 4;
        case MatType::MAT_F32:
            return 4;
        case MatType::MAT_F64:
            return 8;
        default:
            return 0;
    }

    return 0;
} 

class Mat
{
public:
    CUDA_HOST Mat() : m_sizes(0), m_stride(0), m_type(MatType::MAT_INVALID), m_mem(MemType::MEM_INVALID), m_total_bytes(0),
                                m_raw_data(nullptr), m_data(nullptr), m_ref_count(nullptr), m_mat_all_on_cuda(nullptr) {}

    CUDA_HOST Mat(const Sizes& sizes, MatType type, MemType mem = MemType::MEM_CPU, const Sizes& strides = Sizes());

    // shallow copy
    CUDA_HOST Mat(const Mat& other_mat);

    // Shallow copy assignment operator
    CUDA_HOST Mat& operator=(const Mat& other_mat);

    /**
     * @brief clone the matrix.
     * 
     * the stride of the new matrix is the same as the old one.
     * 
     * @param mem Memory type of the new matrix. If MemType::MEM_INVALID, the new matrix will use the same memory type as the old one.
    */
    CUDA_HOST Mat clone(MemType mem = MemType::MEM_INVALID) const;

    CUDA_HOST_DEVICE inline bool empty() const { return 0 == m_total_bytes || nullptr == m_raw_data; }

    CUDA_HOST_DEVICE inline bool equalSizeAndType(const Mat& other_mat) const
    {
        return m_sizes.equal(other_mat.m_sizes) && m_type == other_mat.m_type && m_mem == other_mat.m_mem;
    }

    /**
     * @param row row index
     * @param col column index
     * @param ch  channel index
    */
    template<typename Tp> inline
    CUDA_HOST_DEVICE const Tp* ptr(int32_t row = 0, int32_t col = 0, int32_t ch = 0) const
    {
        return (Tp *)(m_data + row * m_stride.m_w + col * m_sizes.m_ch * MatTypeSize(m_type) + ch * MatTypeSize(m_type));
    }

    /** @overload
     * @param row row index
     * @param col column index
     * @param ch  channel index
    */
    template<typename Tp> inline
    CUDA_HOST_DEVICE Tp* ptr(int32_t row = 0, int32_t col = 0, int32_t ch = 0)
    {
        return (Tp *)(m_data + row * m_stride.m_w + col * m_sizes.m_ch * MatTypeSize(m_type) + ch * MatTypeSize(m_type));
    }

    /**
     * @param row row index
     * @param col column index
     * @param ch  channel index
    */
    template<typename Tp> inline
    CUDA_HOST_DEVICE const Tp& at(int32_t row, int32_t col, int32_t ch = 0) const
    {
        return *ptr<Tp>(row, col, ch);
    }

    /** @overload
     * @param row row index
     * @param col column index
     * @param ch  channel index
    */
    template<typename Tp> inline
    CUDA_HOST_DEVICE Tp& at(int32_t row, int32_t col, int32_t ch = 0)
    {
        return *ptr<Tp>(row, col, ch);
    }

    /**
     * @brief randomize the matrix.
    */
    template<typename Tp>
    CUDA_HOST_DEVICE void random()
    {
        std::random_device rd;

        std::mt19937 gen(rd());

        std::uniform_real_distribution<> dis(0, m_sizes.m_w);

        for (int32_t i = 0; i < m_sizes.m_h; ++i)
        {
            for (int32_t j = 0; j < m_sizes.m_w; ++j)
            {
                for (int32_t k = 0; k < m_sizes.m_ch; ++k)
                {
                    at<Tp>(i, j, k) = (int32_t)dis(gen);
                }
            }
        }
    }

    template<typename Tp>
    CUDA_HOST_DEVICE int32_t compare(const Mat& other_mat) const
    {
        if (!equalSizeAndType(other_mat))
        {
            LOG_ERROR("Mat::compare: size and type not equal");
            return RET_ERR;
        }

        for (int32_t y = 0; y < m_sizes.m_h; ++y)
        {
            for (int32_t x = 0; x < m_sizes.m_w; ++x)
            {
                for (int32_t c = 0; c < m_sizes.m_ch; ++c)
                {
                    auto diff = std::abs(at<Tp>(y, x, c) - other_mat.at<Tp>(y, x, c));

                    if (std::is_floating_point<Tp>::value)
                    {
                        if (diff > 1e-5)
                        {
                            LOG_ERROR("Mat::compare: diff is too large: %f:%f, y/x/c: (%d*%d*%d)", this->at<Tp>(y, x, c),
                                      other_mat.at<Tp>(y, x, c), y, x, c);
                            return RET_ERR;
                        }
                    }
                    else
                    {
                        if (diff > 0)
                        {
                            LOG_ERROR("Mat::compare: diff is too large: %d:%d, y/x/c: (%d*%d*%d)", (int32_t)this->at<Tp>(y, x, c),
                                     (int32_t)other_mat.at<Tp>(y, x, c), y, x, c);
                            return RET_ERR;
                        }
                    }
                }

            }
        }

        LOG_INFO("**** Cmp Ok****");
        return RET_OK;
    }

   CUDA_HOST_DEVICE Mat *GetMatAllOnCUDAMem();

    CUDA_HOST ~Mat();

    Sizes   m_sizes;                    // size of the matrix
    Sizes   m_stride;                   // stride of the matrix
    MatType m_type;                     // type of the matrix
    MemType m_mem;                      // memory type of the matrix
    int32_t m_total_bytes;              // total size of the matrix

private:
    // increase reference count by delta, return the new reference count
    CUDA_HOST int32_t AddReference(int32_t delta = 1);

    CUDA_HOST void Release();

    uint8_t *m_raw_data;                // raw data of the matrix
    uint8_t *m_data;                    // data of the matrix
    std::atomic<int32_t> *m_ref_count;  // reference count of the matrix

    Mat *m_mat_all_on_cuda = nullptr;               // the memory of header and data all are allocated on CUDA 
};

} // namespace cudaup