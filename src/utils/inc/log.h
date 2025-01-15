#pragma once

#include <iostream>
#include <sstream>

// 定义颜色的 ANSI 转义序列
#define COLOR_RESET "\033[0m"
#define COLOR_DEBUG "\033[32m"   // 绿色
#define COLOR_INFO "\033[34m"    // 蓝色
#define COLOR_ERROR "\033[31m"   // 红

#define LOG_INFO(fmt, ...) \
    std::printf(COLOR_INFO "[CUDAUp] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_DEBUG(fmt, ...) \
    std::printf(COLOR_DEBUG "[CUDAUp] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
    std::printf(COLOR_ERROR "[CUDAUp] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define RET_OK 0
#define RET_ERR 1