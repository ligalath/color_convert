/**
 * @file main.cpp
 * @author dongjunchen
 * @brief 
 * @version 0.1
 * @date 2023-09-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils_cuda.h"
#include "utils_log.h"
extern void BGR2J420P_gpu(const unsigned char* src_data_gpu, const int src_stride, unsigned char* dst_data_gpu, int dst_y_stride, int width, int height);
int main()
{
    cv::Mat bgr_mat = cv::imread("test.jpg", cv::IMREAD_UNCHANGED);
    if(!InitCUDA())
    {
        return -1;
    }
    unsigned char* yuv_gpu = nullptr;
    unsigned char* bgr_gpu = nullptr;
    TIC(cudaMalloc);
    cudaMalloc((void**)&yuv_gpu, sizeof(unsigned char)* bgr_mat.cols * bgr_mat.rows * 3/2);
    cudaMalloc((void**)&bgr_gpu, sizeof(unsigned char)* bgr_mat.cols * bgr_mat.rows * 3);
    TOC(cudaMalloc);
    TIC(convert);
    TIC(cudaMemcpy1);
    cudaMemcpy(bgr_gpu, bgr_mat.data, sizeof(unsigned char) * bgr_mat.rows * bgr_mat.step, cudaMemcpyHostToDevice);
    TOC(cudaMemcpy1);
    int y_stride = bgr_mat.cols;
    BGR2J420P_gpu(bgr_gpu, bgr_mat.cols*3, yuv_gpu, y_stride, bgr_mat.cols, bgr_mat.rows);

    std::unique_ptr<unsigned char[]> yuv_cpu(new unsigned char[bgr_mat.cols * bgr_mat.rows * 3/2]);
    unsigned char* yuv_cpu_ptr = yuv_cpu.get();
    TIC(cudaMemcpy2);
    cudaMemcpy(yuv_cpu_ptr, yuv_gpu, sizeof(unsigned char) * bgr_mat.cols * bgr_mat.rows * 3/2, cudaMemcpyDeviceToHost);
    TOC(cudaMemcpy2);
    TOC(convert);
    cudaFree(yuv_gpu);
    cudaFree(bgr_gpu);

    //write
    FILE* fp_yuv = fopen("./out.yuv", "wb");
    fwrite(yuv_cpu.get(), 1, bgr_mat.cols * bgr_mat.rows * 3/2, fp_yuv);
    fclose(fp_yuv);
    return 1;
}