#pragma once

#include "utils_cuda.cuh"

#ifdef D3Q19 //              0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 
    const ci_t H_CIX[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
    const ci_t H_CIY[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
    const ci_t H_CIZ[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
    constexpr float W_0 = 1.0f / 3.0f;  // 0
    constexpr float W_1 = 1.0f / 18.0f; // 1 to 6
    constexpr float W_2 = 1.0f / 36.0f; // 7 to 18
    const float H_W[19] = { 1.0f / 3.0f, 
                            1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
                            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 
                            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
    constexpr int FLINKS = 19;
#elif defined(D3Q27) //      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
    const ci_t H_CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
    const ci_t H_CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
    const ci_t H_CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
    constexpr float W_0 = 8.0f / 27.0f;  // 0
    constexpr float W_1 = 2.0f / 27.0f;  // 1 to 6
    constexpr float W_2 = 1.0f / 54.0f;  // 7 to 18
    constexpr float W_3 = 1.0f / 216.0f; // 19 to 26
    const float H_W[27] = { 8.0f / 27.0f,
                            2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
                            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
                            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
                            1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f };
    constexpr int FLINKS = 27;
#endif

constexpr float W_G_1 = 1.0f / 4.0f; // 0
constexpr float W_G_2 = 1.0f / 8.0f; // 1 to 6
const float H_W_G[7] = { 1.0f / 4.0f, 
                         1.0f / 8.0f, 1.0f / 8.0f, 
                         1.0f / 8.0f, 1.0f / 8.0f, 
                         1.0f / 8.0f, 1.0f / 8.0f };
constexpr int GLINKS = 7;   
