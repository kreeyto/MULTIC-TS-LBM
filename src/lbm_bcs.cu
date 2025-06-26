#include "kernels.cuh"

#define INFLOW_CASE_THREE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    float center_x = (NX-1) * 0.5f;
    float center_y = (NY-1) * 0.5f;

    float dx = x-center_x, dy = y-center_y;
    float radial_dist = sqrtf(dx*dx + dy*dy);
    float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    #ifdef INFLOW_CASE_ONE 
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
        float phi_in = profile * envelope; 
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
        #else
            float uz_in = U_JET * phi_in;
        #endif
    #elif defined(INFLOW_CASE_TWO)
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        float phi_in = 1.0f;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * envelope;
        #else
            float uz_in = U_JET * envelope;
        #endif
    #elif defined(INFLOW_CASE_THREE) 
        float phi_in = 1.0f;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
        #else
            float uz_in = U_JET
        #endif
    #endif

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val; // copy density from the inside
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,0.0f,uz_in,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}


/*

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool is_valid_edge = (x < NX && y < NY && z < NZ) &&
                            (x == 0 || x == NX-1 ||
                             y == 0 || y == NY-1 || 
                                       z == NZ-1); // prevent writes on inflow
    if (!is_valid_edge) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);
    const float rho_val = d.rho[idx3];
    const float phi_val = d.phi[idx3];
    const float ux_val = d.ux[idx3];
    const float uy_val = d.uy[idx3];
    const float uz_val = d.uz[idx3];
    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);

    // extrapolate fneq from the inside of the domain

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        int xo = x + CIX[OPP[Q]];
        int yo = y + CIY[OPP[Q]];
        int zo = z + CIZ[OPP[Q]];
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xo >= 0 && xo < NX && yo >= 0 && yo < NY && zo >= 0 && zo < NZ) {
            idx_t idx4_in = gpu_idx_global4(xo, yo, zo, Q);
            idx_t idx4_out = gpu_idx_global4(xx, yy, zz, Q);

            float rho_in = d.rho[gpu_idx_global3(xo, yo, zo)];
            float ux_in  = d.ux [gpu_idx_global3(xo, yo, zo)];
            float uy_in  = d.uy [gpu_idx_global3(xo, yo, zo)];
            float uz_in  = d.uz [gpu_idx_global3(xo, yo, zo)];
            float uu_in  = 1.5f * (ux_in*ux_in + uy_in*uy_in + uz_in*uz_in);

            float fi_in = d.f[idx4_in];
            float feq_in = gpu_compute_equilibria(rho_in, ux_in, uy_in, uz_in, uu_in, Q);
            float fneq_in = fi_in - feq_in;

            float feq_out = gpu_compute_equilibria(rho_val, ux_val, uy_val, uz_val, uu, Q);
            d.f[idx4_out] = feq_out + (1.0f - OMEGA) * fneq_in;
        }
    }
    
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const idx_t idx4 = gpu_idx_global4(x,y,z,Q);
        float geq = gpu_compute_truncated_equilibria(phi_val,ux_val,uy_val,uz_val,Q);
        d.g[idx4] = geq;
    }
}
*/

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
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
            d.f[streamed_idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
            d.g[streamed_idx4] = W_G[Q] * d.phi[idx3] - W_G[Q];
        }
    }
}


// ============================================================================================================== //

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

    d.phi[gpu_idx_global3(x,y,z)] = d.phi[gpu_idx_global3(x,y,z-1)];
    d.rho[gpu_idx_global3(x,y,z)] = d.rho[gpu_idx_global3(x,y,z-1)];
    d.ux[gpu_idx_global3(x,y,z)] = d.ux[gpu_idx_global3(x,y,z-1)];
    d.uy[gpu_idx_global3(x,y,z)] = d.uy[gpu_idx_global3(x,y,z-1)];
    d.uz[gpu_idx_global3(x,y,z)] = d.uz[gpu_idx_global3(x,y,z-1)];
}

