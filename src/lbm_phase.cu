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

    float pop[GLINKS];
    pop[0] = d.g[gpu_idx_global4(x,y,z,0)];
    pop[1] = d.g[gpu_idx_global4(x,y,z,1)];
    pop[2] = d.g[gpu_idx_global4(x,y,z,2)];
    pop[3] = d.g[gpu_idx_global4(x,y,z,3)];
    pop[4] = d.g[gpu_idx_global4(x,y,z,4)];
    pop[5] = d.g[gpu_idx_global4(x,y,z,5)];
    pop[6] = d.g[gpu_idx_global4(x,y,z,6)];

    const float phi_val = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6];
        
    d.phi[idx3] = phi_val;
}

//#elif defined(D3Q27) //      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
//    const ci_t H_CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
//    const ci_t H_CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
//    const ci_t H_CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

__global__ void gpuGradients(LBMFields d) {
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
    
    const float grad2 = grad_phi_x*grad_phi_x + grad_phi_y*grad_phi_y + grad_phi_z*grad_phi_z;
    const float mag = rsqrtf(grad2 + 1e-9f);
    const float normx_val = grad_phi_x * mag;
    const float normy_val = grad_phi_y * mag;
    const float normz_val = grad_phi_z * mag;

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

    const float phi_val = d.phi[idx3];
    const float normx_val = d.normx[idx3];
    const float normy_val = d.normy[idx3];
    const float normz_val = d.normz[idx3];
    const float ind_val = phi_val * (1.0f - phi_val) * (normx_val*normx_val + normy_val*normy_val + normz_val*normz_val);

    float curvature = 0.0f;
    if (ind_val > 0.2f) {
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
        curvature = -3.0f * (w_sum_curv_x + w_sum_curv_y + w_sum_curv_z);   
    }

    const float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx_val;
    d.ffy[idx3] = coeff_force * normy_val;
    d.ffz[idx3] = coeff_force * normz_val;
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
