#include "kernels.cuh"

__global__ void gpuMomCollisionStream(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);
        
    float fneq[FLINKS];
    float pop[FLINKS];
      
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        pop[Q] = d.f[idx4];
    }

    float rho_val = 0.0f;
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) 
        rho_val += pop[Q];

    float inv_rho = 1.0f / rho_val;

    #ifdef D3Q19
        float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8]  + pop[9]  - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
        float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8]  + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
        float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);
    #elif defined(D3Q27)
        float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8]  + pop[9]  - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
        float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8]  + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
        float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);
    #endif

    float ffx_val = d.ffx[idx3];
    float ffy_val = d.ffy[idx3];
    float ffz_val = d.ffz[idx3];

    float fx_corr = ffx_val * 0.5f * inv_rho;
    float fy_corr = ffy_val * 0.5f * inv_rho;
    float fz_corr = ffz_val * 0.5f * inv_rho;

    float ux_val = sum_ux + fx_corr;
    float uy_val = sum_uy + fy_corr;
    float uz_val = sum_uz + fz_corr;

    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);
    float inv_rho_cssq = 3.0f * inv_rho;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        float pre_feq = gpuComputeEquilibriaSecondOrder(rho_val,ux_val,uy_val,uz_val,uu,Q);
        float he_force = COEFF_HE * pre_feq * ((CIX[Q] - ux_val) * ffx_val +
                                               (CIY[Q] - uy_val) * ffy_val +
                                               (CIZ[Q] - uz_val) * ffz_val) * inv_rho_cssq;
        float feq = pre_feq - he_force; 
        fneq[Q] = pop[Q] - feq;
    }

    float PXX = fneq[1]  + fneq[2]  + fneq[7]  + fneq[8]  + fneq[9]  + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3]  + fneq[4]  + fneq[7]  + fneq[8]  + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5]  + fneq[6]  + fneq[9]  + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7]  - fneq[13] + fneq[8]  - fneq[14];
    float PXZ = fneq[9]  - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];
    #ifdef D3Q27
    PXX += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PYY += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PZZ += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PXY += fneq[19] - fneq[23] + fneq[20] - fneq[24] + fneq[21] - fneq[25] + fneq[22] - fneq[26];
    PXZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[23] - fneq[25] + fneq[24] - fneq[26];
    PYZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[25] - fneq[23] + fneq[26] - fneq[24];
    #endif // D3Q27
 
    d.ux[idx3] = ux_val; d.uy[idx3] = uy_val; d.uz[idx3] = uz_val;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpuComputeEquilibriaSecondOrder(rho_val,ux_val,uy_val,uz_val,uu,Q);
        float he_force = COEFF_HE * feq * ( (CIX[Q] - ux_val) * ffx_val +
                                            (CIY[Q] - uy_val) * ffy_val +
                                            (CIZ[Q] - uz_val) * ffz_val ) * inv_rho_cssq;
        float fneq_scalar = (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * PXX +
                                             (CIY[Q]*CIY[Q] - CSSQ) * PYY +
                                             (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ +
                                             2.0f * CIX[Q] * CIY[Q] * PXY +
                                             2.0f * CIX[Q] * CIZ[Q] * PXZ +
                                             2.0f * CIY[Q] * CIZ[Q] * PYZ);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.f[streamed_idx4] = feq + (1.0f - OMEGA) * fneq_scalar + he_force; 
    }
}

__global__ void gpuEvolvePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float ux_val = d.ux[idx3];
    float uy_val = d.uy[idx3];
    float uz_val = d.uz[idx3];
    float phi_val = d.phi[idx3];
    float normx_val = d.normx[idx3]; 
    float normy_val = d.normy[idx3];
    float normz_val = d.normz[idx3];

    float phi_norm = GAMMA * (1.0f - phi_val);
    float phi_term_x = 3.0f * ux_val + phi_norm * normx_val;
    float phi_term_y = 3.0f * uy_val + phi_norm * normy_val;
    float phi_term_z = 3.0f * uz_val + phi_norm * normz_val;
    d.g[gpuIdxGlobal4(x,y,z,0)]   = W_G[0] * phi_val;
    d.g[gpuIdxGlobal4(x+1,y,z,1)] = W_G[1] * phi_val * (1.0f + phi_term_x);
    d.g[gpuIdxGlobal4(x-1,y,z,2)] = W_G[2] * phi_val * (1.0f - phi_term_x);
    d.g[gpuIdxGlobal4(x,y+1,z,3)] = W_G[3] * phi_val * (1.0f + phi_term_y);
    d.g[gpuIdxGlobal4(x,y-1,z,4)] = W_G[4] * phi_val * (1.0f - phi_term_y);
    d.g[gpuIdxGlobal4(x,y,z+1,5)] = W_G[5] * phi_val * (1.0f + phi_term_z);
    d.g[gpuIdxGlobal4(x,y,z-1,6)] = W_G[6] * phi_val * (1.0f - phi_term_z);
    
}


