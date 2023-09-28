/**
 * @file utils_cuda.h
 * @author dongjunchen
 * @brief 
 * @version 0.1
 * @date 2023-09-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <cuda_runtime.h>

inline bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(0 == count)
    {
        std::cout << "could not find gpu\n";
        return false;
    }
    int device_id = 0;
    for(int device_id = 0; device_id < count; ++device_id)
    {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, device_id) == cudaSuccess)
        {
            std::cout <<"major: " << prop.major << "\n";
            if(prop.major >= 1)
            {
                std::cout<< "Max Grid siz_x:" << prop.maxGridSize[0]
                         << "\nMax Grid size_y:" << prop.maxGridSize[1]
                         << "\nMax Grid size_z:" << prop.maxGridSize[2]
                         << "\nMax Threads per Block:" << prop.maxThreadsPerBlock << std::endl;
                break;
            }
        }
    }
    if(device_id == count)
    {
        std::cout << "there is no device supporting CUDA 11.x\n";
        return false;
    }
    cudaSetDevice(device_id);
    return true;
}
class cudaDeleter{
public:
    void operator()(void *x)
    {
        cudaFree(x);
    }
};
