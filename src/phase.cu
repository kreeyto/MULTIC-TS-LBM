#include "kernels.cuh"

__global__ void gpuEvolvePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;
    __shared__ float s_phi[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];
    __shared__ float s_normx[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];
    __shared__ float s_normy[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];
    __shared__ float s_normz[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];

    float pop[GLINKS];
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q)
        pop[Q] = d.g[gpuIdxGlobal4(x,y,z,Q)];

    float ux_val = d.ux[idx3];
    float uy_val = d.uy[idx3];
    float uz_val = d.uz[idx3];
    float phi_val = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6]) + 1.0f;
        
    s_phi[lz][ly][lx] = phi_val;
    if (threadIdx.x == 0)              s_phi[lz][ly][lx-1] = d.phi[gpuIdxGlobal3(x-1,y,z)];
    if (threadIdx.x == BLOCK_SIZE_X-1) s_phi[lz][ly][lx+1] = d.phi[gpuIdxGlobal3(x+1,y,z)];
    if (threadIdx.y == 0)              s_phi[lz][ly-1][lx] = d.phi[gpuIdxGlobal3(x,y-1,z)];
    if (threadIdx.y == BLOCK_SIZE_Y-1) s_phi[lz][ly+1][lx] = d.phi[gpuIdxGlobal3(x,y+1,z)];
    if (threadIdx.z == 0)              s_phi[lz-1][ly][lx] = d.phi[gpuIdxGlobal3(x,y,z-1)];
    if (threadIdx.z == BLOCK_SIZE_Z-1) s_phi[lz+1][ly][lx] = d.phi[gpuIdxGlobal3(x,y,z+1)];
    __syncthreads();

    float gradx = 0.375f * (s_phi[lz][ly][lx+1] - s_phi[lz][ly][lx-1]);
    float grady = 0.375f * (s_phi[lz][ly+1][lx] - s_phi[lz][ly-1][lx]);
    float gradz = 0.375f * (s_phi[lz+1][ly][lx] - s_phi[lz-1][ly][lx]);

    float grad2 = gradx*gradx + grady*grady + gradz*gradz;
    float mag = rsqrtf(grad2 + 1e-6f);
    float normx_val = gradx * mag;
    float normy_val = grady * mag;
    float normz_val = gradz * mag;

    s_normx[lz][ly][lx] = normx_val;
    s_normy[lz][ly][lx] = normy_val;
    s_normz[lz][ly][lx] = normz_val;
    if (threadIdx.x == 0) {
        s_normx[lz][ly][lx-1] = d.normx[gpuIdxGlobal3(x-1,y,z)];
        s_normy[lz][ly][lx-1] = d.normy[gpuIdxGlobal3(x-1,y,z)];
        s_normz[lz][ly][lx-1] = d.normz[gpuIdxGlobal3(x-1,y,z)];
    }
    if (threadIdx.x == BLOCK_SIZE_X-1) {
        s_normx[lz][ly][lx+1] = d.normx[gpuIdxGlobal3(x+1,y,z)];
        s_normy[lz][ly][lx+1] = d.normy[gpuIdxGlobal3(x+1,y,z)];
        s_normz[lz][ly][lx+1] = d.normz[gpuIdxGlobal3(x+1,y,z)];
    }
    if (threadIdx.y == 0) {
        s_normx[lz][ly-1][lx] = d.normx[gpuIdxGlobal3(x,y-1,z)];
        s_normy[lz][ly-1][lx] = d.normy[gpuIdxGlobal3(x,y-1,z)];
        s_normz[lz][ly-1][lx] = d.normz[gpuIdxGlobal3(x,y-1,z)];
    }
    if (threadIdx.y == BLOCK_SIZE_Y-1) {
        s_normx[lz][ly+1][lx] = d.normx[gpuIdxGlobal3(x,y+1,z)];
        s_normy[lz][ly+1][lx] = d.normy[gpuIdxGlobal3(x,y+1,z)];
        s_normz[lz][ly+1][lx] = d.normz[gpuIdxGlobal3(x,y+1,z)];
    }
    if (threadIdx.z == 0) {
        s_normx[lz-1][ly][lx] = d.normx[gpuIdxGlobal3(x,y,z-1)];
        s_normy[lz-1][ly][lx] = d.normy[gpuIdxGlobal3(x,y,z-1)];
        s_normz[lz-1][ly][lx] = d.normz[gpuIdxGlobal3(x,y,z-1)];
    }
    if (threadIdx.z == BLOCK_SIZE_Z-1) {
        s_normx[lz+1][ly][lx] = d.normx[gpuIdxGlobal3(x,y,z+1)];
        s_normy[lz+1][ly][lx] = d.normy[gpuIdxGlobal3(x,y,z+1)];
        s_normz[lz+1][ly][lx] = d.normz[gpuIdxGlobal3(x,y,z+1)];
    }
    __syncthreads();

    float curvature = -0.375f * (s_normx[lz][ly][lx+1] - s_normx[lz][ly][lx-1] +
                                 s_normy[lz][ly+1][lx] - s_normy[lz][ly-1][lx] +
                                 s_normz[lz+1][ly][lx] - s_normz[lz-1][ly][lx]);
    float ind_val = grad2 * mag;
    float coeff_force = SIGMA * curvature;
    float ffx_val = coeff_force * normx_val * ind_val;
    float ffy_val = coeff_force * normy_val * ind_val;
    float ffz_val = coeff_force * normz_val * ind_val;
    
    float phi_norm = GAMMA * phi_val * (1.0f - phi_val);
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpuComputeTruncatedEquilibria(phi_val,ux_val,uy_val,uz_val,Q) - W_G[Q];
        float anti_diff = W_G[Q] * phi_norm * (CIX[Q] * normx_val + CIY[Q] * normy_val + CIZ[Q] * normz_val);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq + anti_diff;
    }
    
    d.phi[idx3] = phi_val;
    d.ffx[idx3] = ffx_val;
    d.ffy[idx3] = ffy_val;
    d.ffz[idx3] = ffz_val;
}