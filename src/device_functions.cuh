#pragma once
#include "common.cuh"

//#define SECOND_ORDER // no need to def because theres still no 3rd order counterpart for the tensorial equilibria form

__device__ __forceinline__ int gpu_idx_global3(int x, int y, int z) {
    return x + y * NX + z * NX * NY;
}

__device__ __forceinline__ int gpu_idx_global4(int x, int y, int z, int Q) {
    int stride = NX * NY;
    return x + y * NX + z * stride + Q * stride * NZ;
}

/*
__device__ __forceinline__ int gpu_idx_local3(int x, int y, int z) {
    return x + y * BLOCK_SIZE_X + z * BLOCK_SIZE_X * BLOCK_SIZE_Y;
}

__device__ __forceinline__ int gpu_idx_local4(int x, int y, int z, int Q) {
    int stride = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    return x + y * BLOCK_SIZE_X + z * stride + Q * stride * BLOCK_SIZE_Z;
}
*/

__device__ __forceinline__ float gpu_smoothstep(float edge0, float edge1, float x) {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float gpu_compute_truncated_equilibria(float density, float ux, float uy, float uz, int Q) {
    float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

__device__ __forceinline__ float gpu_compute_equilibria(float rho, float ux, float uy, float uz, float uu, int Q) {
    float cxux = CIX[Q] * ux, cyuy = CIY[Q] * uy, czuz = CIZ[Q] * uz;
    float cu = cxux + cyuy + czuz;
    float ciuciub = cxux * cxux + cyuy * cyuy + czuz * czuz +
                    2.0f * (cxux * cyuy + cxux * czuz + cyuy * czuz);
    return W[Q] * rho * (1.0f + 3.0f * cu + 4.5f * ciuciub - uu);
}


