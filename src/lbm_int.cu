#include "kernels.cuh"

__global__ void gpuComputePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);
    
    float pop[GLINKS];

    pop[0] = d.g[gpuIdxGlobal4(x,y,z,0)];
    pop[1] = d.g[gpuIdxGlobal4(x,y,z,1)];
    pop[2] = d.g[gpuIdxGlobal4(x,y,z,2)];
    pop[3] = d.g[gpuIdxGlobal4(x,y,z,3)];
    pop[4] = d.g[gpuIdxGlobal4(x,y,z,4)];
    pop[5] = d.g[gpuIdxGlobal4(x,y,z,5)];
    pop[6] = d.g[gpuIdxGlobal4(x,y,z,6)];

    float phi_pre_shift = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6];
    float phi_val = phi_pre_shift + 1.0f;

    d.phi[idx3] = phi_val;
}

__global__ void gpuComputeGradients(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float grad_phi_x = 0.375f * (d.phi[gpuIdxGlobal3(x+1,y,z)] - d.phi[gpuIdxGlobal3(x-1,y,z)]);
    float grad_phi_y = 0.375f * (d.phi[gpuIdxGlobal3(x,y+1,z)] - d.phi[gpuIdxGlobal3(x,y-1,z)]);
    float grad_phi_z = 0.375f * (d.phi[gpuIdxGlobal3(x,y,z+1)] - d.phi[gpuIdxGlobal3(x,y,z-1)]);

    float squared = grad_phi_x*grad_phi_x + grad_phi_y*grad_phi_y + grad_phi_z*grad_phi_z;
    float mag = rsqrtf(fmaxf(squared,1e-6f));
    float normx_val = grad_phi_x * mag;
    float normy_val = grad_phi_y * mag;
    float normz_val = grad_phi_z * mag;
    float ind_val = squared * mag;

    d.normx[idx3] = normx_val;
    d.normy[idx3] = normy_val;
    d.normz[idx3] = normz_val;
    d.ind[idx3] = ind_val;
}

__global__ void gpuComputeCurvature(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float normx_val = d.normx[idx3];
    float normy_val = d.normy[idx3];
    float normz_val = d.normz[idx3];
    float ind_val = d.ind[idx3];

    float curvature = -0.375f * (d.normx[gpuIdxGlobal3(x+1,y,z)] - d.normx[gpuIdxGlobal3(x-1,y,z)] +
                                 d.normy[gpuIdxGlobal3(x,y+1,z)] - d.normy[gpuIdxGlobal3(x,y-1,z)] +
                                 d.normz[gpuIdxGlobal3(x,y,z+1)] - d.normz[gpuIdxGlobal3(x,y,z-1)]);

    float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx_val * ind_val;
    d.ffy[idx3] = coeff_force * normy_val * ind_val;
    d.ffz[idx3] = coeff_force * normz_val * ind_val;
}
