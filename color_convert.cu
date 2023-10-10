/**
 * @file color_convert.cu
 * @author dongjunchen
 * @brief 
 * @version 0.1
 * @date 2023-09-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdio.h>

//*********************BT601***********************************//
//  Y = 16  + 0.257 * R + 0.504 * g + 0.098 * b                //
// Cb = 128 - 0.148 * R - 0.291 * g + 0.439 * b                //
// Cr = 128 + 0.439 * R - 0.368 * g - 0.071 * b                //
//  R = 1.164 *(Y - 16) + 1.596 *(Cr - 128)                    //
//  G = 1.164 *(Y - 16) - 0.392 *(Cb - 128) - 0.812 *(Cr - 128)//
//  B = 1.164 *(Y - 16) + 2.016 *(Cb - 128)                    //
//*********************BT601***********************************//

/****************************BT709**********************************************///
#define RGB2Y(R,G,B) (16 + 0.183 * R + 0.614 * G + 0.062 * B)                    //
#define RGB2U(R,G,B) (128 - 0.101 * R - 0.339 * G + 0.439 * B)                   //
#define RGB2V(R,G,B) (128 + 0.439 * R - 0.399 * G - 0.040 * B)                   //
#define YUV2R(Y,U,V) (1.164 *(Y - 16) + 1.792 *(V - 128))                        //
#define YUV2G(Y,U,V) (1.164 *(Y - 16) - 0.213 *(U - 128) - 0.534 *(V - 128))     //
#define YUV2B(Y,U,V) (1.164 *(Y - 16) + 2.114 *(U - 128))                        //
/****************************BT709**********************************************///

#define CLIPVALUE(x, min, max) ((x) < (min)?(min):((x) > (max) ? (max) : (x)))
/// @brief 1D grid 1D block
/// @param src_data 
/// @param src_stride 
/// @param dst_data 
/// @param dst_y_stride 
/// @param width 
/// @param height 
/// @return 
__global__ void BGR2YUVJ420P(const unsigned char* src_data, const int src_stride, unsigned char* dst_data, int dst_y_stride, int width, int height)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    for(int i = thread_id; i < width * height; i += grid_stride)
    {
        int col = i % width;
        int row = i / width;
        if(col >= width || row >= height)
            return;
        
        unsigned char* dst_y = dst_data;
        unsigned char* dst_u = dst_y + height * dst_y_stride;
        unsigned char* dst_v = dst_u + height * dst_y_stride/4;
        unsigned char b = src_data[row * src_stride + col * 3 + 0];
        unsigned char g = src_data[row * src_stride + col * 3 + 1];
        unsigned char r = src_data[row * src_stride + col * 3 + 2];

        dst_y[row * dst_y_stride + col] = (unsigned char)(CLIPVALUE(RGB2Y(r,g,b), 0, 255));
        int uv_stride = dst_y_stride/2;
        int uv_col = col/2;
        int uv_row = row/2;
        if(col % 2 == 0 && row  % 2 == 0)
        {
            dst_u[uv_row * uv_stride + uv_col] = (unsigned char)(CLIPVALUE(RGB2U(r,g,b), 0, 255));
            dst_v[uv_row * uv_stride + uv_col] = (unsigned char)(CLIPVALUE(RGB2V(r,g,b), 0, 255));
        }
    }
    return;
}

/// @brief convert bgra to yuva420P(color_range:jpeg)1D grid 1D block
/// @param src_data 
/// @param src_stride 
/// @param dst_data 
/// @param dst_y_stride 
/// @param width 
/// @param height 
/// @return 
__global__ void BGRA2YUVAJ420P(const unsigned char* src_data, const int src_stride, unsigned char* dst_data, int dst_y_stride, int width, int height)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    for(int i = thread_id; i < width* height; i+=grid_stride)
    {
        int col = i % width;
        int row = i / width;
        unsigned char* dst_a = dst_data + height * dst_y_stride * 3/2;
        unsigned char a = src_data[row * src_stride + col * 3 + 3];
        dst_a[row * dst_y_stride + col] = (unsigned char)a;
    }
    BGR2YUVJ420P(src_data,src_stride,dst_data,dst_y_stride,width,height);
}

__global__ void YUV420P2BGR(const unsigned char* src_data, const int src_y_stride, unsigned char* dst_data, int dst_stride, int width, int height)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    for(int i = thread_id; i < width*height; i+=grid_stride)
    {
        int col = i % width;
        int row = i / width;
        unsigned char* dst_b = dst_data + row * dst_stride + col*3 + 0;
        unsigned char* dst_g = dst_data + row * dst_stride + col*3 + 1;
        unsigned char* dst_r = dst_data + row * dst_stride + col*3 + 2;
        const unsigned char* dst_u = src_data + height * src_y_stride;
        const unsigned char* dst_v = src_data + height * src_y_stride/4;
        const int src_uv_stride = src_y_stride/4;
        const int uv_row = row/2;
        const int uv_col = col/2;
        unsigned char y = src_data[row * src_y_stride + col];
        unsigned char u = dst_u[uv_row *src_uv_stride + uv_col];
        unsigned char v = dst_v[uv_row *src_uv_stride + uv_col];
        *dst_b = (unsigned char)(CLIPVALUE(YUV2B(y, u, v), 0, 255));
        *dst_g = (unsigned char)(CLIPVALUE(YUV2G(y, u, v), 0, 255));
        *dst_r = (unsigned char)(CLIPVALUE(YUV2R(y, u, v), 0, 255));
    }
    return;
}
// extern "C" void RGB2J420P_cpu(const unsigned char* src_data, const int src_stride, unsigned char* dst_data, int dst_y_stride, int width, int height)
// {
//     unsigned char* yuv_gpu = nullptr;
//     unsigned char* rgb_gpu = nullptr;
//     cudaMalloc((void**)&yuv_gpu, sizeof(unsigned char)* rgb_mat.cols * rgb_mat.rows * 3/2);
//     cudaMalloc((void**)&rgb_gpu, sizeof(unsigned char)* rgb_mat.cols * rgb_mat.rows * 3);
//     cudaMemcpy(rgb_gpu, rgb_mat.data, sizeof(unsigned char) * rgb_mat.rows * rgb_mat.step);
// }

bool BGR2J420P_gpu(const unsigned char* src_data_gpu, const int src_stride, unsigned char* dst_data_gpu, int dst_y_stride, int width, int height)
{
    //width + height must be even
    if(width % 2 != 0 || height % 2 != 0)
    {
        fprintf(stderr, "%s: image width and height must be even\n", __FUNCTION__);
        return false;
    }
    const int max_threads_per_block = 1024;
    const int max_blocks_per_grid = 2000;
    int threads_per_block = width > max_threads_per_block?max_threads_per_block:width;
    int blocks_per_grid = height > max_blocks_per_grid? max_blocks_per_grid:height;
    BGR2YUVJ420P<<<blocks_per_grid,threads_per_block>>> (src_data_gpu, src_stride, dst_data_gpu, dst_y_stride, width, height);
    cudaDeviceSynchronize();
    
    cudaError_t cuda_stat = cudaGetLastError();
    if(cudaSuccess != cuda_stat)
    {
        fprintf(stderr,"%s launch failed : %s .\n",__FUNCTION__, cudaGetErrorString(cuda_stat));
        return false;
    }
    return true;
}