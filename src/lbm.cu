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

    const float phi_val = d.g[gpu_idx_global4(x,y,z,0)] + d.g[gpu_idx_global4(x,y,z,1)] + 
                          d.g[gpu_idx_global4(x,y,z,2)] + d.g[gpu_idx_global4(x,y,z,3)] + 
                          d.g[gpu_idx_global4(x,y,z,4)] + d.g[gpu_idx_global4(x,y,z,5)] + 
                          d.g[gpu_idx_global4(x,y,z,6)];
        
    d.phi[idx3] = phi_val;
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
    const float normx_val = grad_phi_x / (ind_val + 1e-9f);
    const float normy_val = grad_phi_y / (ind_val + 1e-9f);
    const float normz_val = grad_phi_z / (ind_val + 1e-9f);

    d.ind[idx3] = ind_val;
    d.normx[idx3] = normx_val;
    d.normy[idx3] = normy_val;
    d.normz[idx3] = normz_val;
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

    const float normx_val = d.normx[idx3];
    const float normy_val = d.normy[idx3];
    const float normz_val = d.normz[idx3];
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
    d.ffx[idx3] = coeff_force * normx_val * ind_val;
    d.ffy[idx3] = coeff_force * normy_val * ind_val;
    d.ffz[idx3] = coeff_force * normz_val * ind_val;
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
    pop[0]  = from_dtype(d.f[gpu_idx_global4(x,y,z,0)]);
    pop[1]  = from_dtype(d.f[gpu_idx_global4(x,y,z,1)]);
    pop[2]  = from_dtype(d.f[gpu_idx_global4(x,y,z,2)]);
    pop[3]  = from_dtype(d.f[gpu_idx_global4(x,y,z,3)]);
    pop[4]  = from_dtype(d.f[gpu_idx_global4(x,y,z,4)]);
    pop[5]  = from_dtype(d.f[gpu_idx_global4(x,y,z,5)]); 
    pop[6]  = from_dtype(d.f[gpu_idx_global4(x,y,z,6)]);
    pop[7]  = from_dtype(d.f[gpu_idx_global4(x,y,z,7)]);
    pop[8]  = from_dtype(d.f[gpu_idx_global4(x,y,z,8)]);
    pop[9]  = from_dtype(d.f[gpu_idx_global4(x,y,z,9)]);
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
        const float rho_val = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18]) + 1.0f;
    #elif defined(D3Q27)
        const float rho_val = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) + 1.0f;
    #endif
    d.rho[idx3] = rho_val;

    const float inv_rho = 1.0f / rho_val;
    const float ffx_val = d.ffx[idx3];
    const float ffy_val = d.ffy[idx3];
    const float ffz_val = d.ffz[idx3];

    #ifdef D3Q19
        const float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
        const float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
        const float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);
    #elif defined(D3Q27)
        const float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
        const float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8]  + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
        const float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);
    #endif

    const float fx_corr = ffx_val * 0.5f * inv_rho;
    const float fy_corr = ffy_val * 0.5f * inv_rho;
    const float fz_corr = ffz_val * 0.5f * inv_rho;

    const float ux_val = sum_ux + fx_corr;
    const float uy_val = sum_uy + fy_corr;
    const float uz_val = sum_uz + fz_corr;

    d.ux[idx3] = ux_val; 
    d.uy[idx3] = uy_val; 
    d.uz[idx3] = uz_val;

    const float inv_rho_cssq = 3.0f * inv_rho;
    const float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);

    #ifdef D3Q19
    float feq = W_1 * (rho_val + rho_val * (-uu + ux_val * (3.0f + 4.5f*ux_val))) - W_1;
    #elif defined(D3Q27)
    float feq = W_1 * (rho_val + rho_val * (-uu + ux_val * (3.0f + ux_val * (4.5f + 4.5f*ux_val) - 3.0f*uu))) - W_1;
    #endif
    float fneq = pop[1] - feq;
    float PXX = fneq;

    #ifdef D3Q19
    feq = W_1 * (rho_val + rho_val * (-uu - ux_val * (3.0f - 4.5f*ux_val))) - W_1;
    #elif defined(D3Q27)
    feq = W_1 * (rho_val + rho_val * (-uu - ux_val * (3.0f - ux_val * (4.5f - 4.5f*ux_val) - 3.0f*uu))) - W_1;
    #endif
    fneq = pop[2] - feq;
    PXX += fneq;

    #ifdef D3Q19
    feq = W_1 * (rho_val + rho_val * (-uu + uy_val * (3.0f + 4.5f*uy_val))) - W_1;
    #elif defined(D3Q27)
    feq = W_1 * (rho_val + rho_val * (-uu + uy_val * (3.0f + uy_val * (4.5f + 4.5f*uy_val) - 3.0f*uu))) - W_1;
    #endif
    fneq = pop[3] - feq;
    float PYY = fneq;

    #ifdef D3Q19
    feq = W_1 * (rho_val + rho_val * (-uu - uy_val * (3.0f - 4.5f*uy_val))) - W_1;
    #elif defined(D3Q27)
    feq = W_1 * (rho_val + rho_val * (-uu - uy_val * (3.0f - uy_val * (4.5f - 4.5f*uy_val) - 3.0f*uu))) - W_1;
    #endif
    fneq = pop[4] - feq;
    PYY += fneq;

    #ifdef D3Q19
    feq = W_1 * (rho_val + rho_val * (-uu + uz_val * (3.0f + 4.5f*uz_val))) - W_1;
    #elif defined(D3Q27)
    feq = W_1 * (rho_val + rho_val * (-uu + uz_val * (3.0f + uz_val * (4.5f + 4.5f*uz_val) - 3.0f*uu))) - W_1;
    #endif
    fneq = pop[5] - feq;
    float PZZ = fneq;

    #ifdef D3Q19
    feq = W_1 * (rho_val + rho_val * (-uu - uz_val * (3.0f - 4.5f*uz_val))) - W_1;
    #elif defined(D3Q27)
    feq = W_1 * (rho_val + rho_val * (-uu - uz_val * (3.0f - uz_val * (4.5f - 4.5f*uz_val) - 3.0f*uu))) - W_1;
    #endif
    fneq = pop[6] - feq;
    PZZ += fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val + uy_val) * (3.0f + 4.5f*(ux_val + uy_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val + uy_val) * (3.0f + (ux_val + uy_val) * (4.5f + 4.5f*(ux_val + uy_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[7] - feq;
    PXX += fneq; PYY += fneq; float PXY = fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu - (ux_val + uy_val) * (3.0f - 4.5f*(ux_val + uy_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu - (ux_val + uy_val) * (3.0f - (ux_val + uy_val) * (4.5f - 4.5f*(ux_val + uy_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[8] - feq;
    PXX += fneq; PYY += fneq; PXY += fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val + uz_val) * (3.0f + 4.5f*(ux_val + uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val + uz_val) * (3.0f + (ux_val + uz_val) * (4.5f + 4.5f*(ux_val + uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[9] - feq;
    PXX += fneq; PZZ += fneq; float PXZ = fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu - (ux_val + uz_val) * (3.0f - 4.5f*(ux_val + uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu - (ux_val + uz_val) * (3.0f - (ux_val + uz_val) * (4.5f - 4.5f*(ux_val + uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[10] - feq;
    PXX += fneq; PZZ += fneq; PXZ += fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val + uz_val) * (3.0f + 4.5f*(uy_val + uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val + uz_val) * (3.0f + (uy_val + uz_val) * (4.5f + 4.5f*(uy_val + uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[11] - feq;
    PYY += fneq; PZZ += fneq; float PYZ = fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu - (uy_val + uz_val) * (3.0f - 4.5f*(uy_val + uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu - (uy_val + uz_val) * (3.0f - (uy_val + uz_val) * (4.5f - 4.5f*(uy_val + uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[12] - feq;
    PYY += fneq; PZZ += fneq; PYZ += fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val - uy_val) * (3.0f + 4.5f*(ux_val - uy_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val - uy_val) * (3.0f + (ux_val - uy_val) * (4.5f + 4.5f*(ux_val - uy_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[13] - feq;
    PXX += fneq; PYY += fneq; PXY -= fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val - ux_val) * (3.0f + 4.5f*(uy_val - ux_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val - ux_val) * (3.0f + (uy_val - ux_val) * (4.5f + 4.5f*(uy_val - ux_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[14] - feq;
    PXX += fneq; PYY += fneq; PXY -= fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val - uz_val) * (3.0f + 4.5f*(ux_val - uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (ux_val - uz_val) * (3.0f + (ux_val - uz_val) * (4.5f + 4.5f*(ux_val - uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[15] - feq;
    PXX += fneq; PZZ += fneq; PXZ -= fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (uz_val - ux_val) * (3.0f + 4.5f*(uz_val - ux_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (uz_val - ux_val) * (3.0f + (uz_val - ux_val) * (4.5f + 4.5f*(uz_val - ux_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[16] - feq;
    PXX += fneq; PZZ += fneq; PXZ -= fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val - uz_val) * (3.0f + 4.5f*(uy_val - uz_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (uy_val - uz_val) * (3.0f + (uy_val - uz_val) * (4.5f + 4.5f*(uy_val - uz_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[17] - feq;
    PYY += fneq; PZZ += fneq; PYZ -= fneq;

    #ifdef D3Q19
    feq = W_2 * (rho_val + rho_val * (-uu + (uz_val - uy_val) * (3.0f + 4.5f*(uz_val - uy_val)))) - W_2;
    #elif defined(D3Q27)
    feq = W_2 * (rho_val + rho_val * (-uu + (uz_val - uy_val) * (3.0f + (uz_val - uy_val) * (4.5f + 4.5f*(uz_val - uy_val)) - 3.0f*uu))) - W_2;
    #endif
    fneq = pop[18] - feq;
    PYY += fneq; PZZ += fneq; PYZ -= fneq;
    
    #ifdef D3Q27
    feq = W_3 * (rho_val + rho_val * (-uu + (ux_val + uy_val + uz_val) * (3.0f + (ux_val + uy_val + uz_val) * (4.5f + 4.5f*(ux_val + uy_val + uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[19] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY += fneq; PXZ += fneq; PYZ += fneq;

    feq = W_3 * (rho_val + rho_val * (-uu - (ux_val + uy_val + uz_val) * (3.0f - (ux_val + uy_val + uz_val) * (4.5f - 4.5f*(ux_val + uy_val + uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[20] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY += fneq; PXZ += fneq; PYZ += fneq;

    feq = W_3 * (rho_val + rho_val * (-uu + (ux_val + uy_val - uz_val) * (3.0f + (ux_val + uy_val - uz_val) * (4.5f + 4.5f*(ux_val + uy_val - uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[21] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY += fneq; PXZ -= fneq; PYZ -= fneq;

    feq = W_3 * (rho_val + rho_val * (-uu + (uz_val - uy_val - ux_val) * (3.0f + (uz_val - uy_val - ux_val) * (4.5f + 4.5f*(uz_val - uy_val - ux_val)) - 3.0f*uu))) - W_3;
    fneq = pop[22] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY += fneq; PXZ -= fneq; PYZ -= fneq; 

    feq = W_3 * (rho_val + rho_val * (-uu + (ux_val - uy_val + uz_val) * (3.0f + (ux_val - uy_val + uz_val) * (4.5f + 4.5f*(ux_val - uy_val + uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[23] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY -= fneq; PXZ += fneq; PYZ -= fneq;

    feq = W_3 * (rho_val + rho_val * (-uu + (uy_val - ux_val - uz_val) * (3.0f + (uy_val - ux_val - uz_val) * (4.5f + 4.5f*(uy_val - ux_val - uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[24] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY -= fneq; PXZ += fneq; PYZ -= fneq;

    feq = W_3 * (rho_val + rho_val * (-uu + (uy_val - ux_val + uz_val) * (3.0f + (uy_val - ux_val + uz_val) * (4.5f + 4.5f*(uy_val - ux_val + uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[25] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY -= fneq; PXZ -= fneq; PYZ += fneq;

    feq = W_3 * (rho_val + rho_val * (-uu + (ux_val - uy_val - uz_val) * (3.0f + (ux_val - uy_val - uz_val) * (4.5f + 4.5f*(ux_val - uy_val - uz_val)) - 3.0f*uu))) - W_3;
    fneq = pop[26] - feq;
    PXX += fneq; PYY += fneq; PZZ += fneq;
    PXY -= fneq; PXZ -= fneq; PYZ += fneq;
    #endif // D3Q27

    d.pxx[idx3] = PXX;
    d.pyy[idx3] = PYY;
    d.pzz[idx3] = PZZ;
    d.pxy[idx3] = PXY;
    d.pxz[idx3] = PXZ;   
    d.pyz[idx3] = PYZ;

    const float omega_loc = gpu_local_omega(z);
    const float omco_loc = 1.0f - omega_loc;
    const float coeff_force = 1.0f - 0.5f * omega_loc;

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,0);
    float force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,0);
    float fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,0);
    d.f[gpu_idx_global4(x,y,z,0)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,1);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,1);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,1);
    d.f[gpu_idx_global4(x+1,y,z,1)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,2);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,2);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,2);
    d.f[gpu_idx_global4(x-1,y,z,2)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,3);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,3);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,3);
    d.f[gpu_idx_global4(x,y+1,z,3)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,4);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,4);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,4);
    d.f[gpu_idx_global4(x,y-1,z,4)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,5);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,5);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,5);
    d.f[gpu_idx_global4(x,y,z+1,5)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,6);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,6);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,6);
    d.f[gpu_idx_global4(x,y,z-1,6)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,7);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,7);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,7);
    d.f[gpu_idx_global4(x+1,y+1,z,7)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,8);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,8);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,8);
    d.f[gpu_idx_global4(x-1,y-1,z,8)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,9);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,9);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,9);
    d.f[gpu_idx_global4(x+1,y,z+1,9)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,10);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,10);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,10);
    d.f[gpu_idx_global4(x-1,y,z-1,10)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,11);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,11);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,11);
    d.f[gpu_idx_global4(x,y+1,z+1,11)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,12);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,12);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,12);
    d.f[gpu_idx_global4(x,y-1,z-1,12)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,13);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,13);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,13);
    d.f[gpu_idx_global4(x+1,y-1,z,13)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,14);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,14);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,14);
    d.f[gpu_idx_global4(x-1,y+1,z,14)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,15);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,15);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,15);
    d.f[gpu_idx_global4(x+1,y,z-1,15)] = to_dtype(feq + omco_loc * fneq_reg + force_corr); 

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,16);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,16);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,16);
    d.f[gpu_idx_global4(x-1,y,z+1,16)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,17);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,17);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,17);
    d.f[gpu_idx_global4(x,y+1,z-1,17)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,18);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,18);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,18);
    d.f[gpu_idx_global4(x,y-1,z+1,18)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    #ifdef D3Q27
    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,19);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,19);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,19);
    d.f[gpu_idx_global4(x+1,y+1,z+1,19)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,20);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,20);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,20);
    d.f[gpu_idx_global4(x-1,y-1,z-1,20)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,21);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,21);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,21);
    d.f[gpu_idx_global4(x+1,y+1,z-1,21)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,22);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,22);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,22);
    d.f[gpu_idx_global4(x-1,y-1,z+1,22)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);    

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,23);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,23);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,23);
    d.f[gpu_idx_global4(x+1,y-1,z+1,23)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,24);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,24);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,24);
    d.f[gpu_idx_global4(x-1,y+1,z-1,24)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);

    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,25);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,25);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,25);
    d.f[gpu_idx_global4(x-1,y+1,z+1,25)] = to_dtype(feq + omco_loc * fneq_reg + force_corr);    
    
    feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,26);
    force_corr = gpu_compute_force_term(coeff_force,feq,ux_val,uy_val,uz_val,ffx_val,ffy_val,ffz_val,inv_rho_cssq,26);
    fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,26);
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

    const float phi_val = d.phi[idx3];
    const float ux_val = d.ux[idx3];
    const float uy_val = d.uy[idx3];
    const float uz_val = d.uz[idx3];
    const float normx_val = d.normx[idx3];
    const float normy_val = d.normy[idx3];
    const float normz_val = d.normz[idx3];

    // rest 
    d.g[gpu_idx_global4(x,y,z,0)] = W_G_1 * phi_val;

    // helpers
    const float phi_norm = W_G_2 * GAMMA * phi_val * (1.0f - phi_val);
    const float mult_phi = W_G_2 * phi_val;
    const float a3 = 3.0f * mult_phi;

    // orthogonal 
    float geq = mult_phi + a3 * ux_val;
    float anti_diff = phi_norm * normx_val;
    d.g[gpu_idx_global4(x+1,y,z,1)] = geq + anti_diff;
    
    geq = mult_phi - a3 * ux_val;
    d.g[gpu_idx_global4(x-1,y,z,2)] = geq - anti_diff;

    geq = mult_phi + a3 * uy_val;
    anti_diff = phi_norm * normy_val;
    d.g[gpu_idx_global4(x,y+1,z,3)] = geq + anti_diff;

    geq = mult_phi - a3 * uy_val;
    d.g[gpu_idx_global4(x,y-1,z,4)] = geq - anti_diff;

    geq = mult_phi + a3 * uz_val;
    anti_diff = phi_norm * normz_val;
    d.g[gpu_idx_global4(x,y,z+1,5)] = geq + anti_diff;

    geq = mult_phi - a3 * uz_val;
    d.g[gpu_idx_global4(x,y,z-1,6)] = geq - anti_diff;
} 



