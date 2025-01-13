#pragma once

#include "log.h"
#include "def.h"

#include <cstdint>
#include <algorithm>
#include <atomic>

// #include <cuda_runtime.h>

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
                                m_raw_data(nullptr), m_data(nullptr), m_ref_count(nullptr) {}

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

    CUDA_HOST_DEVICE bool empty() const { return 0 == m_total_bytes || nullptr == m_raw_data; }

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
};

} // namespace cudaup