#pragma once

#include "mat.h"

namespace cudaup
{
int32_t MatMulScalar(Mat &mat0, Mat &mat1, Mat &dst);

int32_t MatMulBlockAndNoLocalMem(Mat &mat0, Mat &mat1, Mat &dst);

int32_t MatMulShared(Mat &mat0, Mat &mat1, Mat &dst);

int32_t MatMulSharedTile1D(Mat &mat0, Mat &mat1, Mat &dst);
}