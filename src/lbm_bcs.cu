#include "kernels.cuh"

#define INFLOW_CASE_THREE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    float center_x = NX * 0.5f;
    float center_y = NY * 0.5f;

    float dx = x-center_x, dy = y-center_y;
    float radial_dist = sqrtf(dx*dx + dy*dy);
    float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    #ifdef INFLOW_CASE_ONE // phi gets a smooth transition and border correction, uz decreases with radial distance. JET DOESN'T FLATTEN -> good behavior
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpuSmoothstep(0.6f, 1.0f, radial_dist_norm);
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

    int idx3_in = gpuIdxGlobal3(x,y,z);
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
        float feq = gpuComputeEquilibria(rho_val,0.0f,0.0f,uz_in,uu,Q) - W[Q];
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.f[streamed_idx4] = __float2half(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpuComputeTruncatedEquilibria(phi_in,0.0f,0.0f,uz_in,Q) - W_G[Q];
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.g[streamed_idx4] = __float2half(geq);
    }
}

// ============================================================================================================== //

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool isValidEdge = (x < NX && y < NY && z < NZ) &&
                       (x == 0 || x == NX-1 || 
                        y == 0 || y == NY-1 || 
                        z == NZ-1); 
    if (!isValidEdge) return;
    const int idx3 = gpuIdxGlobal3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
            d.f[streamed_idx4] = __float2half(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
            d.g[streamed_idx4] = __float2half(W_G[Q] * d.phi[idx3] - W_G[Q]);
        }
    }
}

__global__ void gpuApplyPeriodicXY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    if (x == 0 || x == NX-1) {
        const int from_x = (x == 0) ? NX - 2 : 1;
        const int idx3_target = gpuIdxGlobal3(x, y, z);
        const int idx3_source = gpuIdxGlobal3(from_x, y, z);

        d.phi[idx3_target] = d.phi[idx3_source];
        d.rho[idx3_target] = d.rho[idx3_source];
    }

    if (y == 0 || y == NY-1) {
        const int from_y = (y == 0) ? NY - 2 : 1;
        const int idx3_target = gpuIdxGlobal3(x, y, z);
        const int idx3_source = gpuIdxGlobal3(x, from_y, z);

        d.phi[idx3_target] = d.phi[idx3_source];
        d.rho[idx3_target] = d.rho[idx3_source];
    }
}



__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    d.phi[gpuIdxGlobal3(x,y,z)] = d.phi[gpuIdxGlobal3(x,y,z-1)];
    d.rho[gpuIdxGlobal3(x,y,z)] = d.rho[gpuIdxGlobal3(x,y,z-1)];
}




