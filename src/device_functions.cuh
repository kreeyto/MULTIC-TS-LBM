#pragma once
#include "constants.cuh"

#define SECOND_ORDER 

__device__ __forceinline__ idx_t gpu_idx_global3(const int x, const int y, const int z) {
    return x + y * NX + z * NX * NY;
}

__device__ __forceinline__ idx_t gpu_idx_global4(const int x, const int y, const int z, const int Q) {
    int stride = NX * NY;
    return x + y * NX + z * stride + Q * stride * NZ;
}

__device__ __forceinline__ idx_t gpu_idx_shared3(const int tx, const int ty, const int tz) {
    return tx + ty * TILE_X + tz * TILE_X * TILE_Y;
}

__device__ __forceinline__ float gpu_smoothstep(float edge0, float edge1, float x) {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float gpu_compute_truncated_equilibria(float density, float ux, float uy, float uz, int Q) {
    float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

__device__ __forceinline__ float gpu_compute_equilibria(float density, float ux, float uy, float uz, float uu, int Q) {
    #ifdef SECOND_ORDER
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f * cu*cu - uu);
        return W[Q] * (density + eqbase) - W[Q];
    #elif defined(THIRD_ORDER)
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f*cu*cu - uu + OOS*cu*cu*cu - cu*uu);
        return W[Q] * (density + eqbase) - W[Q];
    #endif
}


