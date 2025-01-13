#pragma once

#include <iostream>
#include <sstream>

#define LOG_INFO(...) \
        do  \
        {   \
            std::ostringstream oss; \
            oss << "[CUDAUp] " << __VA_ARGS__; \
            std::cout << "\033[1;34m" << oss.str() << "\033[0m" << std::endl; \
        } while (0)

#define LOG_DEBUG(...) \
        do  \
        {   \
            std::ostringstream oss; \
            oss << "[CUDAUp] " << __VA_ARGS__; \
            std::cout << "\033[1;32m" << oss.str() << "\033[0m" << std::endl; \
        } while (0)

#define LOG_ERROR(...) \
        do  \
        {   \
            std::ostringstream oss; \
            oss << "[CUDAUp] " << __VA_ARGS__; \
            std::cout << "\033[1;31m" << oss.str() << "\033[0m" << std::endl; \
        } while (0)

#define RET_OK 0
#define RET_ERR 1