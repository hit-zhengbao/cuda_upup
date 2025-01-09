#pragma once

#include <iostream>

#define LOG_INFO(msg)  std::cout << "\033[1;34m" << "[CUDAUp] " << msg << "\033[0m" << std::endl
#define LOG_DEBUG(msg) std::cout << "\033[1;32m" << "[CUDAUp] " << msg << "\033[0m" << std::endl
#define LOG_ERROR(msg) std::cout << "\033[1;31m" << "[CUDAUp] " << msg << "\033[0m" << std::endl