#pragma once
#include "common.cuh"

#define SECOND_ORDER

__device__ __forceinline__ int gpuIdxGlobal3(int x, int y, int z) {
    return x + y * NX + z * NX * NY;
}

__device__ __forceinline__ int gpuIdxGlobal4(int x, int y, int z, int Q) {
    int slice = NX * NY;
    return x + y * NX + z * slice + Q * slice * NZ;
}

__device__ __forceinline__ float gpuSmoothstep(float edge0, float edge1, float x) {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float gpuComputeTruncatedEquilibria(float density, float ux, float uy, float uz, int Q) {
    float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

__device__ __forceinline__ float gpuComputeEquilibria(float density, float ux, float uy, float uz, float uu, int Q) {
    #ifdef SECOND_ORDER
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f * cu*cu - uu);
        return W[Q] * (density + eqbase);
    #elif defined(THIRD_ORDER)
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f*cu*cu - uu + OOS*cu*cu*cu - cu*uu);
        return W[Q] * (density + eqbase);
    #endif
}
