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

#define ALIGN(x, y) ((x+y-1) & ~(y-1))
void pad(unsigned char* src_data, int src_width, unsigned char* dst_data, int align_base = 2)
{
    int align_w = ALIGN(src_width,align_base);
    if(align_w > 0)
    {

    }
}