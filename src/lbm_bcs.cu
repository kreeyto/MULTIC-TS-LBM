#include "kernels.cuh"

#ifdef JET_CASE

#define INFLOW_CASE_THREE

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

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(sx,sy,sz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,0.0f,uz_in,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(sx,sy,sz,Q);
        d.g[streamed_idx4] = geq;
    }
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

    const idx_t boundary_idx3 = gpu_idx_global3(x,y,z);
    const float rho_val = d.rho[boundary_idx3];
    const float phi_val = d.phi[boundary_idx3];
    const float ux_val = d.ux[boundary_idx3];
    const float uy_val = d.uy[boundary_idx3];
    const float uz_val = d.uz[boundary_idx3];
    const float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q],      sy = y + CIY[Q],      sz = z + CIZ[Q];
        const int fx = x + CIX[OPP[Q]], fy = y + CIY[OPP[Q]], fz = z + CIZ[OPP[Q]];
        const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q); 
        const float feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,uu,Q);
        if (fx >= 0 && fx < NX && fy >= 0 && fy < NY && fz >= 0 && fz < NZ && sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t neighbor_fluid_idx3 = gpu_idx_global3(fx,fy,fz); 
            const float fneq_reg = (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * d.pxx[neighbor_fluid_idx3] +
                                                    (CIY[Q]*CIY[Q] - CSSQ) * d.pyy[neighbor_fluid_idx3] +
                                                    (CIZ[Q]*CIZ[Q] - CSSQ) * d.pzz[neighbor_fluid_idx3] +
                                                     2.0f * CIX[Q]*CIY[Q] * d.pxy[neighbor_fluid_idx3] +
                                                     2.0f * CIX[Q]*CIZ[Q] * d.pxz[neighbor_fluid_idx3] +
                                                     2.0f * CIY[Q]*CIZ[Q] * d.pyz[neighbor_fluid_idx3]);
            d.f[streamed_boundary_idx4] = to_dtype(feq + OMC * fneq_reg);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_val,ux_val,uy_val,uz_val,Q);
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.g[streamed_boundary_idx4] = geq;
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

