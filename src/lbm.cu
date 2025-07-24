#include "kernels.cuh"

__global__ void gpuCollisionStream(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    float omega_loc = gpu_local_omega(z);
    const float omco_loc = 1.0f - omega_loc;
        
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

    const float inv_rho_cssq = 3.0f * inv_rho;

    float fneq[FLINKS];
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const float pre_feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,Q);
        const float force_corr = 0.5f * pre_feq * ( (CIX[Q] - ux_val) * ffx_val +
                                                    (CIY[Q] - uy_val) * ffy_val +
                                                    (CIZ[Q] - uz_val) * ffz_val ) * inv_rho_cssq;
        const float feq = pre_feq - force_corr;
        fneq[Q] = pop[Q] - feq;
    } //FOR_EACH_FNEQ; // call unrolled loop

    float PXX = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7] - fneq[13] + fneq[8] - fneq[14];
    float PXZ = fneq[9] - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];
    #ifdef D3Q27
    PXX += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PYY += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PZZ += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PXY += fneq[19] - fneq[23] + fneq[20] - fneq[24] + fneq[21] - fneq[25] + fneq[22] - fneq[26];
    PXZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[23] - fneq[25] + fneq[24] - fneq[26];
    PYZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[25] - fneq[23] + fneq[26] - fneq[24];
    #endif // D3Q27

    d.pxx[idx3] = PXX;
    d.pyy[idx3] = PYY;
    d.pzz[idx3] = PZZ;
    d.pxy[idx3] = PXY;
    d.pxz[idx3] = PXZ;   
    d.pyz[idx3] = PYZ;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        const float feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,Q);
        const float force_corr = 0.5f * feq * ( (CIX[Q] - ux_val) * ffx_val +
                                                (CIY[Q] - uy_val) * ffy_val +
                                                (CIZ[Q] - uz_val) * ffz_val ) * inv_rho_cssq;
        const float fneq_reg = gpu_compute_non_equilibria(PXX,PYY,PZZ,PXY,PXZ,PYZ,ux_val,uy_val,uz_val,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq + omco_loc * fneq_reg + force_corr); 
    } //FOR_EACH_STREAM; // unrolled loop

    // write to global memory
    d.rho[idx3] = rho_val;
    d.ux[idx3] = ux_val; 
    d.uy[idx3] = uy_val; 
    d.uz[idx3] = uz_val;
}



