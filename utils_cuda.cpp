/**
 * @file utils_cuda.cpp
 * @author dongjunchen
 * @brief 
 * @version 0.1
 * @date 2023-09-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "utils_cuda.h"

cudaDeleter::void operator()(void *x);
{
    cudaFree(x);
}