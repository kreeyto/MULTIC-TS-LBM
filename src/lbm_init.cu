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
        //const float dx= x-center_x, dy = y-center_y, dz = z-center_z;
        const float radial_dist = sqrtf(dx*dx + dy*dy + dz*dz);

        const float phi_val = 0.5f + 0.5f * tanhf(2.0f * (RADIUS-radial_dist) / 3.0f);
        d.phi[idx3] = phi_val;
    }
#endif

__global__ void gpuInitFields(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    d.rho[idx3] = 1.0f;
}

__global__ void gpuInitDistributions(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        d.f[gpu_idx_global4(x,y,z,Q)] = gpu_compute_equilibria(d.rho[idx3],d.ux[idx3],d.uy[idx3],d.uz[idx3],Q);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        d.g[gpu_idx_global4(x,y,z,Q)] = gpu_compute_truncated_equilibria(d.phi[idx3],d.ux[idx3],d.uy[idx3],d.uz[idx3],Q);
    }
} 


