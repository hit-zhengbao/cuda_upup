#pragma once

#include "log.h"

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

static int MatTypeSize(MatType type)
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
    Mat() : m_sizes(0), m_stride(0), m_type(MatType::MAT_INVALID), m_mem(MemType::MEM_INVALID), m_total_bytes(0),
            m_raw_data(nullptr), m_data(nullptr), m_ref_count(nullptr) {}

    Mat(const Sizes& sizes, MatType type, MemType mem = MemType::MEM_CPU, const Sizes& strides = Sizes());

    // shallow copy
    Mat(const Mat& other_mat);

    // Shallow copy assignment operator
    Mat& operator=(const Mat& other_mat);

    ~Mat();

private:
    // increase reference count by delta, return the new reference count
    int32_t AddReference(int32_t delta = 1);

    void Release();

    Sizes   m_sizes;                    // size of the matrix
    Sizes   m_stride;                   // stride of the matrix
    MatType m_type;                     // type of the matrix
    MemType m_mem;                      // memory type of the matrix
    int32_t m_total_bytes;              // total size of the matrix
    uint8_t *m_raw_data;                // raw data of the matrix
    uint8_t *m_data;                    // data of the matrix
    std::atomic<int32_t> *m_ref_count;  // reference count of the matrix
};

} // namespace cudaup