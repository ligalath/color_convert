/**
 * @file utils_log.h
 * @author dongjunchen
 * @brief 
 * @version 0.1
 * @date 2023-09-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ST_UTILS_LOG_H_
#define _ST_UTILS_LOG_H_

#include <iostream>
#include <chrono>

#define ENABLE_TRACE_LOG 0
#define ENABLE_TIME_LOG 1

#define LOGD(fmt, ...) {char log_buff[1024]; snprintf(log_buff, 1024, fmt, ##__VA_ARGS__); std::cout << "[Debug] : " << log_buff << std::endl;}
#define LOGI(fmt, ...) {char log_buff[1024]; snprintf(log_buff, 1024, fmt, ##__VA_ARGS__); std::cout << "[Info]  : "  << log_buff << std::endl;}
#define LOGE(fmt, ...) {char log_buff[1024]; snprintf(log_buff, 1024, fmt, ##__VA_ARGS__); std::cout << "[Error] : "  << log_buff << std::endl;}
#if ENABLE_TIME_LOG
#define TIC(X) auto X##_start = std::chrono::high_resolution_clock::now()
#define TOC(X) auto X##_stop = std::chrono::high_resolution_clock::now();std::cout<<"[Profile] : "#X  " Cost:" << std::chrono::duration_cast<std::chrono::milliseconds>(X##_stop-X##_start).count() << " ms" << std::endl

#define TIME_SUM_DECLARE(X) static int64_t X##_sum = 0; 
#define TIME_SUM_PRINT(X) std::cout << "[Profile] : "#X "_sum Cost: " << X##_sum << " ms." << std::endl; X##_sum = 0;

#define TIME_AVG_DECLARE(X) TIME_SUM_DECLARE(X); static int X##_num = 0;
#define TIME_AVG_PRINT(X) std::cout << "[Profile] : "#X "_avg Cost: " << (float)X##_sum/(float)X##_num << " ms.\n"\
                                    << "[Profile] : "#X "_sum Cost: " << X##_sum << " ms." << std::endl;\
                                    X##_num = 0;

#define TIME_SUM_TIC(X) auto X##_start = std::chrono::high_resolution_clock::now();
#define TIME_SUM_TOC(X) \
auto X##_stop = std::chrono::high_resolution_clock::now();\
X##_sum += std::chrono::duration_cast<std::chrono::milliseconds>(X##_stop-X##_start).count();

#define TIME_AVG_TIC(X) auto X##_start = std::chrono::high_resolution_clock::now();
#define TIME_AVG_TOC(X) \
TIME_SUM_TOC(X);\
X##_num +=1;

#else
#define TIC(X)
#define TOC(X)
#define TIME_SUM_DECLARE(X)
#define TIME_SUM_PRINT(X)
#define TIME_AVG_DECLARE(X)
#define TIME_AVG_PRINT(X)
#define TIME_SUM_TIC(X)
#define TIME_SUM_TOC(X)
#define TIME_AVG_TIC(X)
#define TIME_AVG_TOC(X)
#endif

#if ENABLE_TRACE_LOG
#define LOGT(fmt, ...) {char log_buff[1024]; snprintf(log_buff, 1024, fmt, ##__VA_ARGS__); std::cout << "[Trace] : " << log_buff << std::endl;}
#else
#define LOGT(fmt, ...)
#endif
#endif //_ST_UTILS_LOG_H_