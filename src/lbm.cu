#include "kernels.cuh"

__global__ void gpuPhi(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float phi = d.g[gpu_idx_global4(x,y,z,0)] + d.g[gpu_idx_global4(x,y,z,1)] + 
                          d.g[gpu_idx_global4(x,y,z,2)] + d.g[gpu_idx_global4(x,y,z,3)] + 
                          d.g[gpu_idx_global4(x,y,z,4)] + d.g[gpu_idx_global4(x,y,z,5)] + 
                          d.g[gpu_idx_global4(x,y,z,6)];
        
    d.phi[idx3] = phi;
}

__global__ void gpuNormals(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    float w_sum_grad_x = W_1 * (d.phi[gpu_idx_global3(x+1,y,z)]   - d.phi[gpu_idx_global3(x-1,y,z)])  +
                         W_2 * (d.phi[gpu_idx_global3(x+1,y+1,z)] - d.phi[gpu_idx_global3(x-1,y-1,z)] +
                                d.phi[gpu_idx_global3(x+1,y,z+1)] - d.phi[gpu_idx_global3(x-1,y,z-1)] +
                                d.phi[gpu_idx_global3(x+1,y-1,z)] - d.phi[gpu_idx_global3(x-1,y+1,z)] +
                                d.phi[gpu_idx_global3(x+1,y,z-1)] - d.phi[gpu_idx_global3(x-1,y,z+1)]);

    float w_sum_grad_y = W_1 * (d.phi[gpu_idx_global3(x,y+1,z)]   - d.phi[gpu_idx_global3(x,y-1,z)])  +
                         W_2 * (d.phi[gpu_idx_global3(x+1,y+1,z)] - d.phi[gpu_idx_global3(x-1,y-1,z)] +
                                d.phi[gpu_idx_global3(x,y+1,z+1)] - d.phi[gpu_idx_global3(x,y-1,z-1)] +
                                d.phi[gpu_idx_global3(x-1,y+1,z)] - d.phi[gpu_idx_global3(x+1,y-1,z)] +
                                d.phi[gpu_idx_global3(x,y+1,z-1)] - d.phi[gpu_idx_global3(x,y-1,z+1)]);

    float w_sum_grad_z = W_1 * (d.phi[gpu_idx_global3(x,y,z+1)]   - d.phi[gpu_idx_global3(x,y,z-1)])  +
                         W_2 * (d.phi[gpu_idx_global3(x+1,y,z+1)] - d.phi[gpu_idx_global3(x-1,y,z-1)] +
                                d.phi[gpu_idx_global3(x,y+1,z+1)] - d.phi[gpu_idx_global3(x,y-1,z-1)] +
                                d.phi[gpu_idx_global3(x-1,y,z+1)] - d.phi[gpu_idx_global3(x+1,y,z-1)] +
                                d.phi[gpu_idx_global3(x,y-1,z+1)] - d.phi[gpu_idx_global3(x,y+1,z-1)]);
    #ifdef D3Q27
    w_sum_grad_x += W_3 * (d.phi[gpu_idx_global3(x+1,y+1,z+1)] - d.phi[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.phi[gpu_idx_global3(x+1,y+1,z-1)] - d.phi[gpu_idx_global3(x-1,y-1,z+1)] +
                           d.phi[gpu_idx_global3(x+1,y-1,z+1)] - d.phi[gpu_idx_global3(x-1,y+1,z-1)] +
                           d.phi[gpu_idx_global3(x+1,y-1,z-1)] - d.phi[gpu_idx_global3(x-1,y+1,z+1)]);

    w_sum_grad_y += W_3 * (d.phi[gpu_idx_global3(x+1,y+1,z+1)] - d.phi[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.phi[gpu_idx_global3(x+1,y+1,z-1)] - d.phi[gpu_idx_global3(x-1,y-1,z+1)] +
                           d.phi[gpu_idx_global3(x-1,y+1,z-1)] - d.phi[gpu_idx_global3(x+1,y-1,z+1)] +
                           d.phi[gpu_idx_global3(x-1,y+1,z+1)] - d.phi[gpu_idx_global3(x+1,y-1,z-1)]);

    w_sum_grad_z += W_3 * (d.phi[gpu_idx_global3(x+1,y+1,z+1)] - d.phi[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.phi[gpu_idx_global3(x-1,y-1,z+1)] - d.phi[gpu_idx_global3(x+1,y+1,z-1)] +
                           d.phi[gpu_idx_global3(x+1,y-1,z+1)] - d.phi[gpu_idx_global3(x-1,y+1,z-1)] +
                           d.phi[gpu_idx_global3(x-1,y+1,z+1)] - d.phi[gpu_idx_global3(x+1,y-1,z-1)]);
    #endif // D3Q27
        
    const float grad_phi_x = 3.0f * w_sum_grad_x;
    const float grad_phi_y = 3.0f * w_sum_grad_y;
    const float grad_phi_z = 3.0f * w_sum_grad_z;
    
    const float ind_val = sqrtf(grad_phi_x*grad_phi_x + grad_phi_y*grad_phi_y + grad_phi_z*grad_phi_z);
    const float inv_ind = 1.0f / (ind_val + 1e-9f);
    const float normx = grad_phi_x * inv_ind;
    const float normy = grad_phi_y * inv_ind;
    const float normz = grad_phi_z * inv_ind;

    d.ind[idx3] = ind_val;
    d.normx[idx3] = normx;
    d.normy[idx3] = normy;
    d.normz[idx3] = normz;
}

__global__ void gpuForces(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float normx = d.normx[idx3];
    const float normy = d.normy[idx3];
    const float normz = d.normz[idx3];
    const float ind_val = d.ind[idx3];

    float w_sum_curv_x = W_1 * (d.normx[gpu_idx_global3(x+1,y,z)]   - d.normx[gpu_idx_global3(x-1,y,z)])  +
                         W_2 * (d.normx[gpu_idx_global3(x+1,y+1,z)] - d.normx[gpu_idx_global3(x-1,y-1,z)] +
                                d.normx[gpu_idx_global3(x+1,y,z+1)] - d.normx[gpu_idx_global3(x-1,y,z-1)] +
                                d.normx[gpu_idx_global3(x+1,y-1,z)] - d.normx[gpu_idx_global3(x-1,y+1,z)] +
                                d.normx[gpu_idx_global3(x+1,y,z-1)] - d.normx[gpu_idx_global3(x-1,y,z+1)]);

    float w_sum_curv_y = W_1 * (d.normy[gpu_idx_global3(x,y+1,z)]   - d.normy[gpu_idx_global3(x,y-1,z)])  +
                         W_2 * (d.normy[gpu_idx_global3(x+1,y+1,z)] - d.normy[gpu_idx_global3(x-1,y-1,z)] +
                                d.normy[gpu_idx_global3(x,y+1,z+1)] - d.normy[gpu_idx_global3(x,y-1,z-1)] +
                                d.normy[gpu_idx_global3(x-1,y+1,z)] - d.normy[gpu_idx_global3(x+1,y-1,z)] +
                                d.normy[gpu_idx_global3(x,y+1,z-1)] - d.normy[gpu_idx_global3(x,y-1,z+1)]);

    float w_sum_curv_z = W_1 * (d.normz[gpu_idx_global3(x,y,z+1)]   - d.normz[gpu_idx_global3(x,y,z-1)])  +
                         W_2 * (d.normz[gpu_idx_global3(x+1,y,z+1)] - d.normz[gpu_idx_global3(x-1,y,z-1)] +
                                d.normz[gpu_idx_global3(x,y+1,z+1)] - d.normz[gpu_idx_global3(x,y-1,z-1)] +
                                d.normz[gpu_idx_global3(x-1,y,z+1)] - d.normz[gpu_idx_global3(x+1,y,z-1)] +
                                d.normz[gpu_idx_global3(x,y-1,z+1)] - d.normz[gpu_idx_global3(x,y+1,z-1)]);
    #ifdef D3Q27
    w_sum_curv_x += W_3 * (d.normx[gpu_idx_global3(x+1,y+1,z+1)] - d.normx[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.normx[gpu_idx_global3(x+1,y+1,z-1)] - d.normx[gpu_idx_global3(x-1,y-1,z+1)] +
                           d.normx[gpu_idx_global3(x+1,y-1,z+1)] - d.normx[gpu_idx_global3(x-1,y+1,z-1)] +
                           d.normx[gpu_idx_global3(x+1,y-1,z-1)] - d.normx[gpu_idx_global3(x-1,y+1,z+1)]);

    w_sum_curv_y += W_3 * (d.normy[gpu_idx_global3(x+1,y+1,z+1)] - d.normy[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.normy[gpu_idx_global3(x+1,y+1,z-1)] - d.normy[gpu_idx_global3(x-1,y-1,z+1)] +
                           d.normy[gpu_idx_global3(x-1,y+1,z-1)] - d.normy[gpu_idx_global3(x+1,y-1,z+1)] +
                           d.normy[gpu_idx_global3(x-1,y+1,z+1)] - d.normy[gpu_idx_global3(x+1,y-1,z-1)]);

    w_sum_curv_z += W_3 * (d.normz[gpu_idx_global3(x+1,y+1,z+1)] - d.normz[gpu_idx_global3(x-1,y-1,z-1)] +
                           d.normz[gpu_idx_global3(x-1,y-1,z+1)] - d.normz[gpu_idx_global3(x+1,y+1,z-1)] +
                           d.normz[gpu_idx_global3(x+1,y-1,z+1)] - d.normz[gpu_idx_global3(x-1,y+1,z-1)] +
                           d.normz[gpu_idx_global3(x-1,y+1,z+1)] - d.normz[gpu_idx_global3(x+1,y-1,z-1)]);
    #endif // D3Q27
    float curvature = -3.0f * (w_sum_curv_x + w_sum_curv_y + w_sum_curv_z);   

    const float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx * ind_val;
    d.ffy[idx3] = coeff_force * normy * ind_val;
    d.ffz[idx3] = coeff_force * normz * ind_val;
}

__global__ void gpuCollisionStream(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);
        
    float pop[FLINKS];
    pop[0] = from_dtype(d.f[gpu_idx_global4(x,y,z,0)]);
    pop[1] = from_dtype(d.f[gpu_idx_global4(x,y,z,1)]);
    pop[2] = from_dtype(d.f[gpu_idx_global4(x,y,z,2)]);
    pop[3] = from_dtype(d.f[gpu_idx_global4(x,y,z,3)]);
    pop[4] = from_dtype(d.f[gpu_idx_global4(x,y,z,4)]);
    pop[5] = from_dtype(d.f[gpu_idx_global4(x,y,z,5)]); 
    pop[6] = from_dtype(d.f[gpu_idx_global4(x,y,z,6)]);
    pop[7] = from_dtype(d.f[gpu_idx_global4(x,y,z,7)]);
    pop[8] = from_dtype(d.f[gpu_idx_global4(x,y,z,8)]);
    pop[9] = from_dtype(d.f[gpu_idx_global4(x,y,z,9)]);
    pop[10] = from_dtype(d.f[gpu_idx_global4(x,y,z,10)]);
    pop[11] = from_dtype(d.f[gpu_idx_global4(x,y,z,11)]);
    pop[12] = from_dtype(d.f[gpu_idx_global4(x,y,z,12)]);
    pop[13] = from_dtype(d.f[gpu_idx_global4(x,y,z,13)]);
    pop[14] = from_dtype(d.f[gpu_idx_global4(x,y,z,14)]);
    pop[15] = from_dtype(d.f[gpu_idx_global4(x,y,z,15)]);
    pop[16] = from_dtype(d.f[gpu_idx_global4(x,y,z,16)]);
    pop[17] = from_dtype(d.f[gpu_idx_global4(x,y,z,17)]);
    pop[18] = from_dtype(d.f[gpu_idx_global4(x,y,z,18)]);
    #ifdef D3Q27
    pop[19] = from_dtype(d.f[gpu_idx_global4(x,y,z,19)]);
    pop[20] = from_dtype(d.f[gpu_idx_global4(x,y,z,20)]);
    pop[21] = from_dtype(d.f[gpu_idx_global4(x,y,z,21)]);
    pop[22] = from_dtype(d.f[gpu_idx_global4(x,y,z,22)]);
    pop[23] = from_dtype(d.f[gpu_idx_global4(x,y,z,23)]);
    pop[24] = from_dtype(d.f[gpu_idx_global4(x,y,z,24)]);
    pop[25] = from_dtype(d.f[gpu_idx_global4(x,y,z,25)]);
    pop[26] = from_dtype(d.f[gpu_idx_global4(x,y,z,26)]);
    #endif // D3Q27

    #ifdef D3Q19
    const float rho = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18]) + 1.0f;
    #elif defined(D3Q27)
    const float rho = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) + 1.0f;
    #endif
    d.rho[idx3] = rho;

    const float inv_rho = 1.0f / rho;
    const float ffx = d.ffx[idx3];
    const float ffy = d.ffy[idx3];
    const float ffz = d.ffz[idx3];

    #ifdef D3Q19
    const float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
    const float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
    const float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);
    #elif defined(D3Q27)
    const float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
    const float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8]  + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
    const float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);
    #endif
    
    const float ux = sum_ux + ffx * 0.5f * inv_rho;
    const float uy = sum_uy + ffy * 0.5f * inv_rho;
    const float uz = sum_uz + ffz * 0.5f * inv_rho;

    d.ux[idx3] = ux; 
    d.uy[idx3] = uy; 
    d.uz[idx3] = uz;

    const float inv_rho_cssq = 3.0f * inv_rho;

    // rest
    // #ifdef D3Q19
    // feq = W_0 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz));
    // #elif defined(D3Q27)
    // feq = W_0 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz));
    // #endif

    // FIRST ORDER TERMS
    #ifdef D3Q19
    float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    float fneq = pop[1] - (feq - W_1);
    float pxx = fneq;

    #ifdef D3Q19
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[2] - (feq - W_1);
    pxx += fneq;

    #ifdef D3Q19
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[3] - (feq - W_1);
    float pyy = fneq;

    #ifdef D3Q19
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[4] - (feq - W_1);
    pyy += fneq;

    #ifdef D3Q19
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[5] - (feq - W_1);
    float pzz = fneq;

    #ifdef D3Q19
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[6] - (feq - W_1);
    pzz += fneq;

    // SECOND ORDER TERMS
    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) + 4.5f*(ux + uy)*(ux + uy)*(ux + uy) - 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[7] - (feq - W_2);
    pxx += fneq; 
    pyy += fneq; 
    float pxy = fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) - 4.5f*(ux + uy)*(ux + uy)*(ux + uy) + 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[8] - (feq - W_2);
    pxx += fneq; 
    pyy += fneq; 
    pxy += fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) + 4.5f*(ux + uz)*(ux + uz)*(ux + uz) - 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[9] - (feq - W_2);
    pxx += fneq; 
    pzz += fneq; 
    float pxz = fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) - 4.5f*(ux + uz)*(ux + uz)*(ux + uz) + 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[10] - (feq - W_2);
    pxx += fneq; 
    pzz += fneq; 
    pxz += fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) + 4.5f*(uy + uz)*(uy + uz)*(uy + uz) - 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[11] - (feq - W_2);
    pyy += fneq;
    pzz += fneq; 
    float pyz = fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) - 4.5f*(uy + uz)*(uy + uz)*(uy + uz) + 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[12] - (feq - W_2);
    pyy += fneq; 
    pzz += fneq; 
    pyz += fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy) + 4.5f*(ux - uy)*(ux - uy)*(ux - uy) - 3.0f*(ux - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[13] - (feq - W_2);
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux) + 4.5f*(uy - ux)*(uy - ux)*(uy - ux) - 3.0f*(uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[14] - (feq - W_2);
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz) + 4.5f*(ux - uz)*(ux - uz)*(ux - uz) - 3.0f*(ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[15] - (feq - W_2);
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux) + 4.5f*(uz - ux)*(uz - ux)*(uz - ux) - 3.0f*(uz - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[16] - (feq - W_2);
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz) + 4.5f*(uy - uz)*(uy - uz)*(uy - uz) - 3.0f*(uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[17] - (feq - W_2);
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;

    #ifdef D3Q19
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy) + 4.5f*(uz - uy)*(uz - uy)*(uz - uy) - 3.0f*(uz - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop[18] - (feq - W_2);
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;
    
    // THIRD ORDER TERMS
    #ifdef D3Q27
    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) - 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[19] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) - 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) + 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[20] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz)*(ux + uy - uz) - 3.0f*(ux + uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));    
    fneq = pop[21] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux)*(uz - uy - ux) - 3.0f*(uz - uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[22] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq; 

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz)*(ux - uy + uz) - 3.0f*(ux - uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[23] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz)*(uy - ux - uz) - 3.0f*(uy - ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[24] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz)*(uy - ux + uz) - 3.0f*(uy - ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[25] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz)*(ux - uy - uz) - 3.0f*(ux - uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop[26] - (feq - W_3);
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;
    #endif // D3Q27

    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;   
    d.pyz[idx3] = pyz;

    const float omega_loc = gpu_local_omega(z);
    const float omco_loc = 1.0f - omega_loc;
    const float coeff_force = 1.0f - 0.5f * omega_loc;

    feq = gpu_compute_equilibria(rho,ux,uy,uz,0);
    float force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,0);
    float fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,0);
    d.f[gpu_idx_global4(x,y,z,0)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,1);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,1);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,1);
    d.f[gpu_idx_global4(x+1,y,z,1)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,2);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,2);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,2);
    d.f[gpu_idx_global4(x-1,y,z,2)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,3);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,3);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,3);
    d.f[gpu_idx_global4(x,y+1,z,3)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,4);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,4);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,4);
    d.f[gpu_idx_global4(x,y-1,z,4)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,5);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,5);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,5);
    d.f[gpu_idx_global4(x,y,z+1,5)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,6);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,6);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,6);
    d.f[gpu_idx_global4(x,y,z-1,6)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,7);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,7);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,7);
    d.f[gpu_idx_global4(x+1,y+1,z,7)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,8);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,8);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,8);
    d.f[gpu_idx_global4(x-1,y-1,z,8)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,9);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,9);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,9);
    d.f[gpu_idx_global4(x+1,y,z+1,9)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,10);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,10);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,10);
    d.f[gpu_idx_global4(x-1,y,z-1,10)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,11);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,11);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,11);
    d.f[gpu_idx_global4(x,y+1,z+1,11)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,12);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,12);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,12);
    d.f[gpu_idx_global4(x,y-1,z-1,12)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,13);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,13);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,13);
    d.f[gpu_idx_global4(x+1,y-1,z,13)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,14);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,14);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,14);
    d.f[gpu_idx_global4(x-1,y+1,z,14)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,15);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,15);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,15);
    d.f[gpu_idx_global4(x+1,y,z-1,15)] = to_dtype(feq + omco_loc * fneq_reg + force_corr); 

    feq = gpu_compute_equilibria(rho,ux,uy,uz,16);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,16);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,16);
    d.f[gpu_idx_global4(x-1,y,z+1,16)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,17);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,17);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,17);
    d.f[gpu_idx_global4(x,y+1,z-1,17)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,18);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,18);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,18);
    d.f[gpu_idx_global4(x,y-1,z+1,18)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    #ifdef D3Q27
    feq = gpu_compute_equilibria(rho,ux,uy,uz,19);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,19);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,19);
    d.f[gpu_idx_global4(x+1,y+1,z+1,19)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,20);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,20);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,20);
    d.f[gpu_idx_global4(x-1,y-1,z-1,20)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,21);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,21);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,21);
    d.f[gpu_idx_global4(x+1,y+1,z-1,21)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,22);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,22);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,22);
    d.f[gpu_idx_global4(x-1,y-1,z+1,22)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);    

    feq = gpu_compute_equilibria(rho,ux,uy,uz,23);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,23);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,23);
    d.f[gpu_idx_global4(x+1,y-1,z+1,23)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,24);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,24);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,24);
    d.f[gpu_idx_global4(x-1,y+1,z-1,24)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho,ux,uy,uz,25);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,25);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,25);
    d.f[gpu_idx_global4(x-1,y+1,z+1,25)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);    
    
    feq = gpu_compute_equilibria(rho,ux,uy,uz,26);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux,uy,uz,ffx,ffy,ffz,inv_rho_cssq,26);
    fneq_reg = gpu_compute_non_equilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,26);
    d.f[gpu_idx_global4(x+1,y-1,z-1,26)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);
    #endif // D3Q27
}

__global__ void gpuEvolvePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;
        
    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float phi = d.phi[idx3];
    const float ux = d.ux[idx3];
    const float uy = d.uy[idx3];
    const float uz = d.uz[idx3];
    const float normx = d.normx[idx3];
    const float normy = d.normy[idx3];
    const float normz = d.normz[idx3];

    d.g[gpu_idx_global4(x,y,z,0)] = W_G_1 * phi;

    const float phi_norm = W_G_2 * GAMMA * phi * (1.0f - phi);
    const float mult_phi = W_G_2 * phi;
    const float a3 = 3.0f * mult_phi;

    float geq = mult_phi + a3 * ux;
    float anti_diff = phi_norm * normx;
    d.g[gpu_idx_global4(x+1,y,z,1)] = geq + anti_diff;
    
    geq = mult_phi - a3 * ux;
    d.g[gpu_idx_global4(x-1,y,z,2)] = geq - anti_diff;

    geq = mult_phi + a3 * uy;
    anti_diff = phi_norm * normy;
    d.g[gpu_idx_global4(x,y+1,z,3)] = geq + anti_diff;

    geq = mult_phi - a3 * uy;
    d.g[gpu_idx_global4(x,y-1,z,4)] = geq - anti_diff;

    geq = mult_phi + a3 * uz;
    anti_diff = phi_norm * normz;
    d.g[gpu_idx_global4(x,y,z+1,5)] = geq + anti_diff;

    geq = mult_phi - a3 * uz;
    d.g[gpu_idx_global4(x,y,z-1,6)] = geq - anti_diff;
} 



