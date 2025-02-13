#pragma once

#include "mat.h"

namespace cudaup
{
int32_t ReduceScalar(Mat &mat, int32_t &sum);

int32_t ReduceNative(Mat &mat, int32_t &sum);

int32_t ReduceInterLeaveAddr(Mat &mat, int32_t &sum);

int32_t ReduceBankConflictFree(Mat &mat, int32_t &sum);
} // namespace cudaup