#include "kernels.cuh"

#ifdef JET_CASE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    const float center_x = (NX-1) * 0.5f;
    const float center_y = (NY-1) * 0.5f;

    const float dx = x-center_x, dy = y-center_y;
    const float radial_dist = sqrtf(dx*dx + dy*dy);
    const float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    const float phi_in = 1.0f;
    #ifdef PERTURBATION
        const float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
    #else
        const float uz_in = U_JET;
    #endif

    const float rho_val = 1.0f;
    const float uu = 1.5f * (uz_in * uz_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val; 
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    //#elif defined(D3Q27) //      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
    //constexpr ci_t H_CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
    //constexpr ci_t H_CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
    //constexpr ci_t H_CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

    // x,y,z+1 -> 5
    float feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y,z+1)],0.0f,0.0f,uz_in,uu,5);
    float fneq_reg = (W_1_TO_6 * 4.5f) * (-CSSQ * d.pxx[gpu_idx_global3(x,y,z+1)] +
                                          -CSSQ * d.pyy[gpu_idx_global3(x,y,z+1)] +
                                           CSCO * d.pzz[gpu_idx_global3(x,y,z+1)]);
    d.f[gpu_idx_global4(x,y,z+1,5)] = to_dtype(feq + OMCO * fneq_reg);

    // x+1,y,z+1 -> 9
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x+1,y,z+1)],0.0f,0.0f,uz_in,uu,9);
    fneq_reg = (W_7_TO_18 * 4.5f) * ( CSCO * d.pxx[gpu_idx_global3(x+1,y,z+1)] +
                                     -CSSQ * d.pyy[gpu_idx_global3(x+1,y,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x+1,y,z+1)] +
                                      2.0f * d.pxz[gpu_idx_global3(x+1,y,z+1)]); 
    d.f[gpu_idx_global4(x+1,y,z+1,9)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y+1,z+1 -> 11
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y+1,z+1)],0.0f,0.0f,uz_in,uu,11);
    fneq_reg = (W_7_TO_18 * 4.5f) * (-CSSQ * d.pxx[gpu_idx_global3(x,y+1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x,y+1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x,y+1,z+1)] +
                                      2.0f * d.pyz[gpu_idx_global3(x,y+1,z+1)]);
    d.f[gpu_idx_global4(x,y+1,z+1,11)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y,z+1 -> 16
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x-1,y,z+1)],0.0f,0.0f,uz_in,uu,16);
    fneq_reg = (W_7_TO_18 * 4.5f) * ( CSCO * d.pxx[gpu_idx_global3(x-1,y,z+1)] +
                                     -CSSQ * d.pyy[gpu_idx_global3(x-1,y,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x-1,y,z+1)] -
                                      2.0f * d.pxz[gpu_idx_global3(x-1,y,z+1)]);
    d.f[gpu_idx_global4(x-1,y,z+1,16)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y-1,z+1 -> 18
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y-1,z+1)],0.0f,0.0f,uz_in,uu,18);
    fneq_reg = (W_7_TO_18 * 4.5f) * (-CSSQ * d.pxx[gpu_idx_global3(x,y-1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x,y-1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x,y-1,z+1)] -
                                      2.0f * d.pyz[gpu_idx_global3(x,y-1,z+1)]);
    d.f[gpu_idx_global4(x,y-1,z+1,18)] = to_dtype(feq + OMCO * fneq_reg);

    #ifdef D3Q27
    // x+1,y+1,z+1 -> 19
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x+1,y+1,z+1)],0.0f,0.0f,uz_in,uu,19);
    fneq_reg = (W_19_TO_26 * 4.5f) * (CSCO * d.pxx[gpu_idx_global3(x+1,y+1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x+1,y+1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x+1,y+1,z+1)] +
                                      2.0f * d.pxy[gpu_idx_global3(x+1,y+1,z+1)] +
                                      2.0f * d.pxz[gpu_idx_global3(x+1,y+1,z+1)] +
                                      2.0f * d.pyz[gpu_idx_global3(x+1,y+1,z+1)]);
    d.f[gpu_idx_global4(x+1,y+1,z+1,19)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y-1,z+1 -> 22
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x-1,y-1,z+1)],0.0f,0.0f,uz_in,uu,22);
    fneq_reg = (W_19_TO_26 * 4.5f) * (CSCO * d.pxx[gpu_idx_global3(x-1,y-1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x-1,y-1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x-1,y-1,z+1)] -
                                      2.0f * d.pxy[gpu_idx_global3(x-1,y-1,z+1)] -
                                      2.0f * d.pxz[gpu_idx_global3(x-1,y-1,z+1)] -
                                      2.0f * d.pyz[gpu_idx_global3(x-1,y-1,z+1)]);
    d.f[gpu_idx_global4(x-1,y-1,z+1,22)] = to_dtype(feq + OMCO * fneq_reg);

    // x+1,y-1,z+1 -> 23
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x+1,y-1,z+1)],0.0f,0.0f,uz_in,uu,23);
    fneq_reg = (W_19_TO_26 * 4.5f) * (CSCO * d.pxx[gpu_idx_global3(x+1,y-1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x+1,y-1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x+1,y-1,z+1)] -
                                      2.0f * d.pxy[gpu_idx_global3(x+1,y-1,z+1)] +
                                      2.0f * d.pxz[gpu_idx_global3(x+1,y-1,z+1)] -
                                      2.0f * d.pyz[gpu_idx_global3(x+1,y-1,z+1)]);
    d.f[gpu_idx_global4(x+1,y-1,z+1,23)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y+1,z+1 -> 25
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x-1,y+1,z+1)],0.0f,0.0f,uz_in,uu,25);
    fneq_reg = (W_19_TO_26 * 4.5f) * (CSCO * d.pxx[gpu_idx_global3(x-1,y+1,z+1)] +
                                      CSCO * d.pyy[gpu_idx_global3(x-1,y+1,z+1)] +
                                      CSCO * d.pzz[gpu_idx_global3(x-1,y+1,z+1)] -
                                      2.0f * d.pxy[gpu_idx_global3(x-1,y+1,z+1)] -
                                      2.0f * d.pxz[gpu_idx_global3(x-1,y+1,z+1)] +
                                      2.0f * d.pyz[gpu_idx_global3(x-1,y+1,z+1)]);
    d.f[gpu_idx_global4(x-1,y+1,z+1,25)] = to_dtype(feq + OMCO * fneq_reg);
    #endif // D3Q27

    // x,y,z+1 -> 5
    feq = W_G_1 * phi_in * (1.0f + 3.0f * uz_in);
    d.g[gpu_idx_global4(x,y,z+1,5)] = feq;
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    d.uz[gpu_idx_global3(x,y,z)] = d.uz[gpu_idx_global3(x,y,z-1)];
    const float ux_val = d.ux[gpu_idx_global3(x,y,z)];
    const float uy_val = d.uy[gpu_idx_global3(x,y,z)];
    const float uz_val = d.uz[gpu_idx_global3(x,y,z)];
    const float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);

    // x,y,z-1 -> 6
    float feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y,z-1)],ux_val,uy_val,uz_val,uu,6);
    float fneq_reg = (W_1_TO_6 * 4.5f) * ((CIX[6]*CIX[6] - CSSQ) * d.pxx[gpu_idx_global3(x,y,z-1)] +
                                          (CIY[6]*CIY[6] - CSSQ) * d.pyy[gpu_idx_global3(x,y,z-1)] +
                                          (CIZ[6]*CIZ[6] - CSSQ) * d.pzz[gpu_idx_global3(x,y,z-1)] +
                                            2.0f * CIX[6]*CIY[6] * d.pxy[gpu_idx_global3(x,y,z-1)] +
                                            2.0f * CIX[6]*CIZ[6] * d.pxz[gpu_idx_global3(x,y,z-1)] +
                                            2.0f * CIY[6]*CIZ[6] * d.pyz[gpu_idx_global3(x,y,z-1)]);
    d.f[gpu_idx_global4(x,y,z-1,6)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y,z-1 -> 10
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x-1,y,z-1)],ux_val,uy_val,uz_val,uu,10);
    fneq_reg = (W_7_TO_18 * 4.5f) * ((CIX[10]*CIX[10] - CSSQ) * d.pxx[gpu_idx_global3(x-1,y,z-1)] +
                                     (CIY[10]*CIY[10] - CSSQ) * d.pyy[gpu_idx_global3(x-1,y,z-1)] +
                                     (CIZ[10]*CIZ[10] - CSSQ) * d.pzz[gpu_idx_global3(x-1,y,z-1)] +
                                       2.0f * CIX[10]*CIY[10] * d.pxy[gpu_idx_global3(x-1,y,z-1)] +
                                       2.0f * CIX[10]*CIZ[10] * d.pxz[gpu_idx_global3(x-1,y,z-1)] +
                                       2.0f * CIY[10]*CIZ[10] * d.pyz[gpu_idx_global3(x-1,y,z-1)]);
    d.f[gpu_idx_global4(x-1,y,z-1,10)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y-1,z-1 -> 12
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y-1,z-1)],ux_val,uy_val,uz_val,uu,12);
    fneq_reg = (W_7_TO_18 * 4.5f) * ((CIX[12]*CIX[12] - CSSQ) * d.pxx[gpu_idx_global3(x,y-1,z-1)] +
                                     (CIY[12]*CIY[12] - CSSQ) * d.pyy[gpu_idx_global3(x,y-1,z-1)] +
                                     (CIZ[12]*CIZ[12] - CSSQ) * d.pzz[gpu_idx_global3(x,y-1,z-1)] +
                                       2.0f * CIX[12]*CIY[12] * d.pxy[gpu_idx_global3(x,y-1,z-1)] +
                                       2.0f * CIX[12]*CIZ[12] * d.pxz[gpu_idx_global3(x,y-1,z-1)] +
                                       2.0f * CIY[12]*CIZ[12] * d.pyz[gpu_idx_global3(x,y-1,z-1)]);
    d.f[gpu_idx_global4(x,y-1,z-1,12)] = to_dtype(feq + OMCO * fneq_reg);

    // x+1,y,z-1 -> 15
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x+1,y,z-1)],ux_val,uy_val,uz_val,uu,15);
    fneq_reg = (W_7_TO_18 * 4.5f) * ((CIX[15]*CIX[15] - CSSQ) * d.pxx[gpu_idx_global3(x+1,y,z-1)] +
                                     (CIY[15]*CIY[15] - CSSQ) * d.pyy[gpu_idx_global3(x+1,y,z-1)] +
                                     (CIZ[15]*CIZ[15] - CSSQ) * d.pzz[gpu_idx_global3(x+1,y,z-1)] +
                                       2.0f * CIX[15]*CIY[15] * d.pxy[gpu_idx_global3(x+1,y,z-1)] +
                                       2.0f * CIX[15]*CIZ[15] * d.pxz[gpu_idx_global3(x+1,y,z-1)] +
                                       2.0f * CIY[15]*CIZ[15] * d.pyz[gpu_idx_global3(x+1,y,z-1)]);
    d.f[gpu_idx_global4(x+1,y,z-1,15)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y+1,z-1 -> 17
    feq = gpu_compute_equilibria(d.rho[gpu_idx_global3(x,y+1,z-1)],ux_val,uy_val,uz_val,uu,17);
    fneq_reg = (W_7_TO_18 * 4.5f) * ((CIX[17]*CIX[17] - CSSQ) * d.pxx[gpu_idx_global3(x,y+1,z-1)] +
                                     (CIY[17]*CIY[17] - CSSQ) * d.pyy[gpu_idx_global3(x,y+1,z-1)] +
                                     (CIZ[17]*CIZ[17] - CSSQ) * d.pzz[gpu_idx_global3(x,y+1,z-1)] +
                                       2.0f * CIX[17]*CIY[17] * d.pxy[gpu_idx_global3(x,y+1,z-1)] +
                                       2.0f * CIX[17]*CIZ[17] * d.pxz[gpu_idx_global3(x,y+1,z-1)] +
                                       2.0f * CIY[17]*CIZ[17] * d.pyz[gpu_idx_global3(x,y+1,z-1)]);
    d.f[gpu_idx_global4(x,y+1,z-1,17)] = to_dtype(feq + OMCO * fneq_reg);
}

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool is_valid_edge = (x < NX && y < NY && z < NZ) &&
                            (x == 0 || x == NX-1 ||
                             y == 0 || y == NY-1 || 
                                       z == NZ-1); 
    if (!is_valid_edge) return;                       

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.f[streamed_boundary_idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.g[streamed_boundary_idx4] = W_G[Q] * d.phi[idx3];
        }
    }
}

__global__ void gpuApplyPeriodicXY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx = gpu_idx_global3(x,y,z);

    if (x == 0 || x == NX-1) {
        const int src_x = (x == 0) ? NX - 2 : 1;
        const idx_t idx_src = gpu_idx_global3(src_x,y,z);
        d.phi[idx] = d.phi[idx_src];
        d.rho[idx] = d.rho[idx_src];
        d.ux[idx] = d.ux[idx_src];
        d.uy[idx] = d.uy[idx_src];
        d.uz[idx] = d.uz[idx_src];
    }

    if (y == 0 || y == NY-1) {
        const int src_y = (y == 0) ? NY - 2 : 1;
        const idx_t idx_src = gpu_idx_global3(x,src_y,z);
        d.phi[idx] = d.phi[idx_src];
        d.rho[idx] = d.rho[idx_src];
        d.ux[idx] = d.ux[idx_src];
        d.uy[idx] = d.uy[idx_src];
        d.uz[idx] = d.uz[idx_src];
    }
}

#elif defined(DROPLET_CASE)

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool is_valid_edge = (x < NX && y < NY && z < NZ) &&
                            (x == 0 || x == NX-1 ||
                             y == 0 || y == NY-1 || 
                             z == 0 || z == NZ-1); 
    if (!is_valid_edge) return;                       

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.f[streamed_boundary_idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.g[streamed_boundary_idx4] = W_G[Q] * d.phi[idx3] - W_G[Q];
        }
    }
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    const idx_t top = gpu_idx_global3(x,y,z);
    const idx_t below = gpu_idx_global3(x,y,z-1);

    d.phi[top] = d.phi[below];
    d.rho[top] = d.rho[below];
    d.ux[top]  = d.ux[below];
    d.uy[top]  = d.uy[below];
    d.uz[top]  = d.uz[below];
}

#endif // FLOW_CASE

// ============================================================================================================== //

