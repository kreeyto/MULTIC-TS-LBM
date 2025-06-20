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

    #ifdef INFLOW_CASE_ONE // phi gets a smooth transition and border correction, uz decreases with radial distance. JET DOESN'T FLATTEN -> good behavior
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
        float phi_in = profile * envelope; 
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
        #else
            float uz_in = U_JET * phi_in;
        #endif
    #elif defined(INFLOW_CASE_TWO) // phi gets a smooth transition but uz is constant throughout the inlet. JET FLATTENS -> bad behavior
        float phi_in = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
        #else
            float uz_in = U_JET;
        #endif
    #elif defined(INFLOW_CASE_THREE) // straight forward inflow. JET FLATTENS -> bad behavior
        float phi_in = 1.0f;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
        #else
            float uz_in = U_JET;
        #endif
    #endif

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,uu,Q) - W[Q];
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,0.0f,uz_in,Q) - W_G[Q];
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}

/*

// these inflow bcs should be after the streaming step
// different from the implemented one that runs first before everything else

// zou-he type inflow condition
__global__ void gpuApplyInflow_ZouHe_FG(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int z = 0;  
    if (x >= NX || y >= NY) return;

    float dx = x - 0.5f*(NX-1);
    float dy = y - 0.5f*(NY-1);
    if (sqrtf(dx*dx + dy*dy) > 0.5f*DIAM) return;

    const float ux_w = 0.0f, uy_w = 0.0f, uz_w = U_JET;
    const float phi_w = 1.0f;

    float rho = 0.0f;
    #pragma unroll
    for (int Q = 0; Q < FLINKS; ++Q) {
        int idx = gpu_idx_global4(x,y,z,Q);
        if (CIZ[Q] <= 0) {
            rho += d.f[idx];
        } else {
            rho += 2.0f * d.f[idx];
        }
    }
    rho = rho / (1.0f + uz_w);  

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        if (CIZ[Q] > 0) {
            int Q_opp = opposite[Q];
            int i   = gpu_idx_global4(x,y,z, Q);
            int i_o = gpu_idx_global4(x,y,z, Q_opp);

            float feq_i   = gpu_compute_equilibria(rho, ux_w, uy_w, uz_w, Q);
            float feq_o   = gpu_compute_equilibria(rho, ux_w, uy_w, uz_w, Q_opp);

            d.f[i] = feq_i + (d.f[i_o] - feq_o);
        }
    }

    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        if (CIZ[Q] > 0) {
            int Q_opp = opposite[Q];  
            int i   = gpu_idx_global4(x,y,z, Q);
            int i_o = gpu_idx_global4(x,y,z, Q_opp);

            float geq_i = gpu_compute_truncated_equilibria(phi_w, ux_w, uy_w, uz_w, Q);
            float geq_o = gpu_compute_truncated_equilibria(phi_w, ux_w, uy_w, uz_w, Q_opp);

            d.g[i] = geq_i + (d.g[i_o] - geq_o);
        }
    }

    idx_t idx3 = gpu_idx_global3(x,y,z);
    d.rho[idx3] = rho;
    d.ux [idx3] = ux_w;
    d.uy [idx3] = uy_w;
    d.uz [idx3] = uz_w;
    d.phi[idx3] = phi_w;
}

// montessori et al thread-safe bcs
// inflow non-eq extrapolation BC (grad's reconstruction)
__global__ void gpuApplyInflow_GradExtrap(LBMFields d, int STEP) {
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int z = 0;  
    if (x>=NX || y>=NY) return;

    float dx = x - 0.5f*(NX-1), dy = y - 0.5f*(NY-1);
    if (sqrtf(dx*dx + dy*dy) > 0.5f*DIAM) return;

    const float rho_B = 1.0f;
    const float phi_B = 1.0f;
    const float ux_B  = 0.0f, uy_B = 0.0f, uz_B = U_JET;

    const float omega_f = 1.0f / TAU_F;
    const float omega_g = 1.0f / TAU_G;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        if (CIZ[Q] > 0) {
            int idxB   = gpu_idx_global4(x,y,z,   Q);
            int idxF   = gpu_idx_global4(x,y,z+1, Q);
            int idx3F  = gpu_idx_global3  (x,y,z+1);

            float rhoF = d.rho[idx3F];
            float uxF  = d.ux [idx3F];
            float uyF  = d.uy [idx3F];
            float uzF  = d.uz [idx3F];

            float feqF = gpu_compute_equilibria(rhoF,uxF,uyF,uzF, Q);
            float f_ne = d.f[idxF] - feqF;

            float feqB = gpu_compute_equilibria(rho_B,ux_B,uy_B,uz_B, Q);

            d.f[idxB] = feqB + (1.0f - omega_f)*f_ne;
        }
    }

    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        if (CIZ[Q] > 0) {
            int idxB   = gpu_idx_global4(x,y,z,   Q);
            int idxF   = gpu_idx_global4(x,y,z+1, Q);
            int idx3F  = gpu_idx_global3  (x,y,z+1);

            float phiF = d.phi[idx3F];

            float geqF = gpu_compute_truncated_equilibria(phiF,ux_B,uy_B,uz_B, Q);
            float g_ne = d.g[idxF] - geqF;

            float geqB = gpu_compute_truncated_equilibria(phi_B,ux_B,uy_B,uz_B, Q);

            d.g[idxB] = geqB + (1.0f - omega_g)*g_ne;
        }
    }

    idx_t idx3 = gpu_idx_global3(x,y,z);
    d.rho[idx3] = rho_B;
    d.ux [idx3] = ux_B;
    d.uy [idx3] = uy_B;
    d.uz [idx3] = uz_B;
    d.phi[idx3] = phi_B;
}
*/

// ============================================================================================================== //

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

__global__ void gpuApplyPeriodicXY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx = gpu_idx_global3(x, y, z);

    if (x == 0 || x == NX-1) {
        const int src_x = (x == 0) ? NX - 2 : 1;
        const idx_t idx_src = gpu_idx_global3(src_x,y,z);
        d.phi[idx] = d.phi[idx_src];
    }

    if (y == 0 || y == NY-1) {
        const int src_y = (y == 0) ? NY - 2 : 1;
        const idx_t idx_src = gpu_idx_global3(x,src_y,z);
        d.phi[idx] = d.phi[idx_src];
    }
}


__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    d.phi[gpu_idx_global3(x,y,z)] = d.phi[gpu_idx_global3(x,y,z-1)];
}




