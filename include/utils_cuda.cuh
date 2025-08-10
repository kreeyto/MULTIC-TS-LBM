#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <builtin_types.h>
#include <math.h>

constexpr int BLOCK_SIZE_X = 8;
constexpr int BLOCK_SIZE_Y = 8;
constexpr int BLOCK_SIZE_Z = 8;

constexpr int TILE_X = BLOCK_SIZE_X + 2;
constexpr int TILE_Y = BLOCK_SIZE_Y + 2;
constexpr int TILE_Z = BLOCK_SIZE_Z + 2;

constexpr size_t DYNAMIC_SHARED_SIZE = 0;

#define FP16_COMPRESSION

#ifdef FP16_COMPRESSION
    #include <cuda_fp16.h>
    typedef __half dtype_t;
    #define to_dtype __float2half
    #define from_dtype __half2float
#else
    typedef float dtype_t;
    #define to_dtype(x) (x)
    #define from_dtype(x) (x)
#endif // FP16_COMPRESSION

typedef int ci_t;
typedef uint32_t idx_t;

#define checkCudaErrors(err)      __checkCudaErrors(err, #err, __FILE__, __LINE__)
#define getLastCudaError(msg)     __getLastCudaError(msg, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

inline void __getLastCudaError(const char* const errorMessage, const char* const file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}


