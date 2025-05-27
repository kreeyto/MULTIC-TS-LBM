#include "kernels.cuh"

//#define INFLOW_CASE_ONE // -> smooth envelope * tanh, phase-coupled velocity (read comment in definition)
#define INFLOW_CASE_TWO // -> diffuse phase only, top-hat velocity profile (read comment in definition)
//#define INFLOW_CASE_THREE // -> straight forward inflow, no phase coupling (read comment in definition)

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
    float interface_width = 3.0f; 
    if (radial_dist > radius) return;

    #ifdef INFLOW_CASE_ONE // phi gets a smooth transition and border correction, uz decreases with radial distance. JET DOESN'T FLATTEN -> good behavior
        //float radial_dist_norm = radial_dist / radius;
        //float envelope = 1.0f - gpuSmoothstep(0.6f, 1.0f, radial_dist_norm);
        float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / interface_width);
        float phi_in = profile; // * envelope;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
        #else
            float uz_in = U_JET * phi_in;
        #endif
    #elif defined(INFLOW_CASE_TWO) // phi gets a smooth transition but uz is constant throughout the inlet. JET FLATTENS -> bad behavior
        float phi_in = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / interface_width);
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
        const int dz = CIZ[Q];
        if (z + dz >= 0 && z + dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            float feq = gpuComputeEquilibriaSecondOrder(rho_val,0.0f,0.0f,uz_in,uu,Q) - W[Q];
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
            d.f[streamed_idx4] = feq;
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int dz = CIZ[Q];
        if (z + dz >= 0 && z + dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            float geq = gpuComputeEquilibriaFirstOrder(phi_in,0.0f,0.0f,uz_in,Q) - W_G[Q];
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
            d.g[streamed_idx4] = geq;
        }
    }
}

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool isValidEdge = (x < NX && y < NY && z < NZ) &&
                       (x == 0 || x == NX-1 || 
                        y == 0 || y == NY-1 || 
                        z == 0 || z == NZ-1); 
    if (!isValidEdge) return;
    const int idx3 = gpuIdxGlobal3(x,y,z);

    // f recovery
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
            d.f[streamed_idx4] = (d.rho[idx3] - 1.0f) * W[Q];
        }
    }
    // g recovery
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
            d.g[streamed_idx4] = (d.phi[idx3] - 1.0f) * W_G[Q];
        }
    }
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    const int idx_top = gpuIdxGlobal3(x,y,z);
    const int idx_below = gpuIdxGlobal3(x,y,z-1);
    d.phi[idx_top] = d.phi[idx_below];
}


