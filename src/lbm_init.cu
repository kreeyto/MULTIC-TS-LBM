#include "kernels.cuh"

#ifdef DROPLET_CASE
    __global__ void gpuInitDropletShape(LBMFields d) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= NX || y >= NY || z >= NZ || 
            x == 0 || x == NX-1 || 
            y == 0 || y == NY-1 || 
            z == 0 || z == NZ-1) return;
        const idx_t idx3 = gpu_idx_global3(x,y,z);
        
        const float center_x = (NX-1) * 0.5f;
        const float center_y = (NY-1) * 0.5f;
        const float center_z = (NZ-1) * 0.5f;

        const float dx = (x-center_x) / 2.0f, dy = y-center_y, dz = z-center_z;
        const float radial_dist = sqrtf(dx*dx + dy*dy + dz*dz);
        const float radius = 0.5 * DIAM;

        const float phi_val = 0.5f + 0.5f * tanhf(2.0f * (radius-radial_dist) / 2.0f);
        d.phi[idx3] = phi_val;
    }
#endif

__global__ void gpuInitFieldsAndDistributions(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;
    const idx_t idx3 = gpu_idx_global3(x,y,z);

    d.rho[idx3] = 1.0f;
    d.pxx[idx3] = 1.0f;
    d.pyy[idx3] = 1.0f;
    d.pzz[idx3] = 1.0f;
    d.pxy[idx3] = 1.0f;
    d.pxz[idx3] = 1.0f;
    d.pyz[idx3] = 1.0f;
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const idx_t idx4 = gpu_idx_global4(x,y,z,Q);
        d.f[idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const idx_t idx4 = gpu_idx_global4(x,y,z,Q);
        d.g[idx4] = W_G[Q] * d.phi[idx3] - W_G[Q];
    }
} 

