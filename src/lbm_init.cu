#include "kernels.cuh"

__global__ void gpuInitFieldsAndDistributions(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;
    const idx_t idx3 = gpu_idx_global3(x,y,z);

    d.rho[idx3] = 1.0f;
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
