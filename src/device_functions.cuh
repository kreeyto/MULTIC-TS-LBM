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

__device__ __forceinline__ float gpu_compute_truncated_equilibria(const float density, const float ux, const float uy, const float uz, const int Q) {
    float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

__device__ __forceinline__ float gpu_compute_equilibria(const float density, const float ux, const float uy, const float uz, const int Q) {
    #ifdef D3Q19
        float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f * cu*cu - uu);
        return W[Q] * (density + eqbase) - W[Q];
    #elif defined(D3Q27)
        float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
        float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        float eqbase = density * (cu + 0.5f*cu*cu - uu + OOS*cu*cu*cu - cu*uu);
        return W[Q] * (density + eqbase) - W[Q];
    #endif
}

__device__ __forceinline__ float gpu_local_omega(int z) {
    float zn = float(z) / float(NZ-1);
    float s = (zn > Z_START) ? (zn - Z_START) / SPONGE : 0.0f;
    float ramp = powf(s,P < 0 ? 1.0f : P);  
    return OMEGA * (1.0f - ramp) + OMEGA_MAX * ramp;                    
}

__device__ __forceinline__ float gpu_compute_non_equilibria(const float PXX, const float PYY, const float PZZ, const float PXY, const float PXZ, const float PYZ,  const float ux, const float uy, const float uz, const int Q) {
    #ifdef D3Q19
        return (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * PXX + 
                                (CIY[Q]*CIY[Q] - CSSQ) * PYY + 
                                (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ + 
                                2.0f * CIX[Q] * CIY[Q] * PXY + 
                                2.0f * CIX[Q] * CIZ[Q] * PXZ +
                                2.0f * CIY[Q] * CIZ[Q] * PYZ);
    #elif defined(D3Q27)
        const float fneq2 = (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * PXX + 
                                             (CIY[Q]*CIY[Q] - CSSQ) * PYY + 
                                             (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ + 
                                             2.0f * CIX[Q] * CIY[Q] * PXY + 
                                             2.0f * CIX[Q] * CIZ[Q] * PXZ +
                                             2.0f * CIY[Q] * CIZ[Q] * PYZ);

        float a3_xxx = 3.0f * ux * PXX;
        float a3_yyy = 3.0f * uy * PYY;
        float a3_zzz = 3.0f * uz * PZZ;

        float a3_xxy = PXX * uy + 2.0f * ux * PXY;
        float a3_xxz = PXX * uz + 2.0f * ux * PXZ;
        float a3_xyy = PXY * uy + 2.0f * ux * PYY;
        float a3_yyz = PYY * uz + 2.0f * uy * PYZ;
        float a3_xzz = PXZ * uz + 2.0f * ux * PZZ;
        float a3_yzz = PYZ * uz + 2.0f * uy * PZZ;

        float a3_xyz = PXY * uz + ux * PYZ + uy * PXZ;

        float H3_xxx = CIX[Q]*CIX[Q]*CIX[Q] - 3.0f*CSSQ*CIX[Q];
        float H3_yyy = CIY[Q]*CIY[Q]*CIY[Q] - 3.0f*CSSQ*CIY[Q];
        float H3_zzz = CIZ[Q]*CIZ[Q]*CIZ[Q] - 3.0f*CSSQ*CIZ[Q];

        float H3_xxy = CIX[Q]*CIX[Q]*CIY[Q] - CSSQ*CIY[Q];
        float H3_xxz = CIX[Q]*CIX[Q]*CIZ[Q] - CSSQ*CIZ[Q];
        float H3_xyy = CIX[Q]*CIY[Q]*CIY[Q] - CSSQ*CIX[Q];
        float H3_yyz = CIY[Q]*CIY[Q]*CIZ[Q] - CSSQ*CIZ[Q];
        float H3_xzz = CIX[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIX[Q];
        float H3_yzz = CIY[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIY[Q];

        float H3_xyz = CIX[Q] * CIY[Q] * CIZ[Q];

        const float fneq3 = (W[Q] * 4.5f) * (
            H3_xxx * a3_xxx
            + H3_yyy * a3_yyy
            + H3_zzz * a3_zzz
            + 3.0f * ( H3_xxy * a3_xxy
                    + H3_xxz * a3_xxz
                    + H3_xyy * a3_xyy
                    + H3_yyz * a3_yyz
                    + H3_xzz * a3_xzz
                    + H3_yzz * a3_yzz )
            + 6.0f * H3_xyz * a3_xyz
        );

        return fneq2 + fneq3;
    #endif // D3Q27 
}

