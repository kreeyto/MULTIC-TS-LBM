#include "kernels.cuh"

__global__ void gpuInitFields(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    d.ux[idx3] = 0.0f;
    d.uy[idx3] = 0.0f;
    d.uz[idx3] = 0.0f;
    d.phi[idx3] = 0.0f;
    d.rho[idx3] = 1.0f;
    d.ffx[idx3] = 0.0f * 1e-7f;
    d.ffy[idx3] = 0.0f * 1e-5f;
    d.ffz[idx3] = 0.0f * 1e-5f;
    d.normx[idx3] = 0.0f;
    d.normy[idx3] = 0.0f;
    d.normz[idx3] = 0.0f;
    d.pxx[idx3] = 0.0f;
    d.pyy[idx3] = 0.0f;
    d.pzz[idx3] = 0.0f;
    d.pxy[idx3] = 0.0f;
    d.pxz[idx3] = 0.0f;
    d.pyz[idx3] = 0.0f;
}

__global__ void gpuInitJetShape(LBMFields d) {
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

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.uz[idx3_in] = U_JET;
    d.phi[idx3_in] = 1.0f;
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
        d.g[gpu_idx_global4(x+CIX[Q],y+CIY[Q],z+CIZ[Q],Q)] = gpu_compute_truncated_equilibria(d.phi[idx3],d.ux[idx3],d.uy[idx3],d.uz[idx3],Q);
    }
} 

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

