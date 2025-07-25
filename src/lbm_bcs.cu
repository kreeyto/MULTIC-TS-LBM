#include "kernels.cuh"

#ifdef JET_CASE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
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
    const float phi_in = 1.0f;
    const float uz_in = 
    #ifdef PERTURBATION
        /* apply perturbation */ U_JET * (1.0f + DATAZ[(STEP/MACRO_SAVE)%200] * 10.0f);
    #else
        /* straightforward */ U_JET;
    #endif 

    const float rho_val = 1.0f;
    d.rho[idx3_in] = rho_val; 
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    //#elif defined(D3Q27) //  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
    //const ci_t H_CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
    //const ci_t H_CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
    //const ci_t H_CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

    // x,y,z+1 -> 5
    int neighbor_idx = gpu_idx_global3(x,y,z+1);
    float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,5);
    float fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                                d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                                d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],5);
    d.f[gpu_idx_global4(x,y,z+1,5)] = to_dtype(feq + OMCO * fneq_reg);

    feq = W_G_2 * phi_in * (1.0f + 3.0f * uz_in);
    d.g[gpu_idx_global4(x,y,z+1,5)] = feq;

    // x+1,y,z+1 -> 9
    neighbor_idx = gpu_idx_global3(x+1,y,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,9);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],9);
    d.f[gpu_idx_global4(x+1,y,z+1,9)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y+1,z+1 -> 11
    neighbor_idx = gpu_idx_global3(x,y+1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,11);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],11);
    d.f[gpu_idx_global4(x,y+1,z+1,11)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y,z+1 -> 16
    neighbor_idx = gpu_idx_global3(x-1,y,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,16);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],16);
    d.f[gpu_idx_global4(x-1,y,z+1,16)] = to_dtype(feq + OMCO * fneq_reg);

    // x,y-1,z+1 -> 18
    neighbor_idx = gpu_idx_global3(x,y-1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,18);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],18);
    d.f[gpu_idx_global4(x,y-1,z+1,18)] = to_dtype(feq + OMCO * fneq_reg);

    #ifdef D3Q27
    // x+1,y+1,z+1 -> 19
    neighbor_idx = gpu_idx_global3(x+1,y+1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,19);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],19);
    d.f[gpu_idx_global4(x+1,y+1,z+1,19)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y-1,z+1 -> 22
    neighbor_idx = gpu_idx_global3(x-1,y-1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,22);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],22);
    d.f[gpu_idx_global4(x-1,y-1,z+1,22)] = to_dtype(feq + OMCO * fneq_reg);

    // x+1,y-1,z+1 -> 23
    neighbor_idx = gpu_idx_global3(x+1,y-1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,23);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],23);
    d.f[gpu_idx_global4(x+1,y-1,z+1,23)] = to_dtype(feq + OMCO * fneq_reg);

    // x-1,y+1,z+1 -> 25
    neighbor_idx = gpu_idx_global3(x-1,y+1,z+1);
    feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,25);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],25);
    d.f[gpu_idx_global4(x-1,y+1,z+1,25)] = to_dtype(feq + OMCO * fneq_reg);
    #endif // D3Q27
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    const int idx_outer = gpu_idx_global3(x,y,z);
    const int idx_inner = gpu_idx_global3(x,y,z-1);
    
    d.rho[idx_outer] = d.rho[idx_inner];
    d.phi[idx_outer] = d.phi[idx_inner];
    d.ux[idx_outer] = d.ux[idx_inner];
    d.uy[idx_outer] = d.uy[idx_inner];
    d.uz[idx_outer] = d.uz[idx_inner];

    const float ux_out = d.ux[idx_outer];
    const float uy_out = d.uy[idx_outer];
    const float uz_out = d.uz[idx_outer];

    // x,y,z-1 -> 6
    int neighbor_idx = gpu_idx_global3(x,y,z-1);
    float feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,6);
    float fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                                d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                                d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],6);
    d.f[gpu_idx_global4(x,y,z-1,6)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    feq = W_G_2 * d.phi[neighbor_idx] * (1.0f - 3.0f * uz_out);
    d.g[gpu_idx_global4(x,y,z-1,6)] = feq;

    // x-1,y,z-1 -> 10
    neighbor_idx = gpu_idx_global3(x-1,y,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,10);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],10);
    d.f[gpu_idx_global4(x-1,y,z-1,10)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x,y-1,z-1 -> 12
    neighbor_idx = gpu_idx_global3(x,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,12);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],12);
    d.f[gpu_idx_global4(x,y-1,z-1,12)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x+1,y,z-1 -> 15
    neighbor_idx = gpu_idx_global3(x+1,y,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,15);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],15);
    d.f[gpu_idx_global4(x+1,y,z-1,15)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x,y+1,z-1 -> 17
    neighbor_idx = gpu_idx_global3(x,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,17);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],17);
    d.f[gpu_idx_global4(x,y+1,z-1,17)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    #ifdef D3Q27
    // x-1,y-1,z-1 -> 20
    neighbor_idx = gpu_idx_global3(x-1,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,20);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],20);
    d.f[gpu_idx_global4(x-1,y-1,z-1,20)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x+1,y+1,z-1 -> 21
    neighbor_idx = gpu_idx_global3(x+1,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,21);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],21);
    d.f[gpu_idx_global4(x+1,y+1,z-1,21)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x-1,y+1,z-1 -> 24
    neighbor_idx = gpu_idx_global3(x-1,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,24);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],24);
    d.f[gpu_idx_global4(x-1,y+1,z-1,24)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    // x+1,y-1,z-1 -> 26
    neighbor_idx = gpu_idx_global3(x+1,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,26);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],26);
    d.f[gpu_idx_global4(x+1,y-1,z-1,26)] = to_dtype(feq + OMCO_MAX * fneq_reg);
    #endif // D3Q27
}

__global__ void gpuApplyPeriodic(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;    

    d.f[gpu_idx_global4(1,y,z,1)]     = d.f[gpu_idx_global4(NX-2,y,z,1)];
    d.f[gpu_idx_global4(1,y,z,7)]     = d.f[gpu_idx_global4(NX-2,y,z,7)];
    d.f[gpu_idx_global4(1,y,z,9)]     = d.f[gpu_idx_global4(NX-2,y,z,9)];
    d.f[gpu_idx_global4(1,y,z,13)]    = d.f[gpu_idx_global4(NX-2,y,z,13)];
    d.f[gpu_idx_global4(1,y,z,15)]    = d.f[gpu_idx_global4(NX-2,y,z,15)];
    d.f[gpu_idx_global4(NX-2,y,z,2)]  = d.f[gpu_idx_global4(1,y,z,2)];
    d.f[gpu_idx_global4(NX-2,y,z,8)]  = d.f[gpu_idx_global4(1,y,z,8)];
    d.f[gpu_idx_global4(NX-2,y,z,10)] = d.f[gpu_idx_global4(1,y,z,10)];
    d.f[gpu_idx_global4(NX-2,y,z,14)] = d.f[gpu_idx_global4(1,y,z,14)];
    d.f[gpu_idx_global4(NX-2,y,z,16)] = d.f[gpu_idx_global4(1,y,z,16)];

    d.f[gpu_idx_global4(x,1,z,3)]     = d.f[gpu_idx_global4(x,NY-2,z,3)];
    d.f[gpu_idx_global4(x,1,z,7)]     = d.f[gpu_idx_global4(x,NY-2,z,7)];
    d.f[gpu_idx_global4(x,1,z,11)]    = d.f[gpu_idx_global4(x,NY-2,z,11)];
    d.f[gpu_idx_global4(x,1,z,14)]    = d.f[gpu_idx_global4(x,NY-2,z,14)];
    d.f[gpu_idx_global4(x,1,z,17)]    = d.f[gpu_idx_global4(x,NY-2,z,17)];
    d.f[gpu_idx_global4(x,NY-2,z,4)]  = d.f[gpu_idx_global4(x,1,z,4)];
    d.f[gpu_idx_global4(x,NY-2,z,8)]  = d.f[gpu_idx_global4(x,1,z,8)];
    d.f[gpu_idx_global4(x,NY-2,z,12)] = d.f[gpu_idx_global4(x,1,z,12)];
    d.f[gpu_idx_global4(x,NY-2,z,13)] = d.f[gpu_idx_global4(x,1,z,13)];
    d.f[gpu_idx_global4(x,NY-2,z,18)] = d.f[gpu_idx_global4(x,1,z,18)];
    #ifdef D3Q27
    d.f[gpu_idx_global4(1,y,z,19)]    = d.f[gpu_idx_global4(NX-2,y,z,19)];
    d.f[gpu_idx_global4(1,y,z,21)]    = d.f[gpu_idx_global4(NX-2,y,z,21)];
    d.f[gpu_idx_global4(1,y,z,23)]    = d.f[gpu_idx_global4(NX-2,y,z,23)];
    d.f[gpu_idx_global4(1,y,z,26)]    = d.f[gpu_idx_global4(NX-2,y,z,26)]; 
    d.f[gpu_idx_global4(NX-2,y,z,20)] = d.f[gpu_idx_global4(1,y,z,20)];
    d.f[gpu_idx_global4(NX-2,y,z,22)] = d.f[gpu_idx_global4(1,y,z,22)];
    d.f[gpu_idx_global4(NX-2,y,z,24)] = d.f[gpu_idx_global4(1,y,z,24)];
    d.f[gpu_idx_global4(NX-2,y,z,25)] = d.f[gpu_idx_global4(1,y,z,25)];

    d.f[gpu_idx_global4(x,1,z,19)]    = d.f[gpu_idx_global4(x,NY-2,z,19)];
    d.f[gpu_idx_global4(x,1,z,21)]    = d.f[gpu_idx_global4(x,NY-2,z,21)];
    d.f[gpu_idx_global4(x,1,z,24)]    = d.f[gpu_idx_global4(x,NY-2,z,24)];
    d.f[gpu_idx_global4(x,1,z,25)]    = d.f[gpu_idx_global4(x,NY-2,z,25)];
    d.f[gpu_idx_global4(x,NY-2,z,20)] = d.f[gpu_idx_global4(x,1,z,20)];
    d.f[gpu_idx_global4(x,NY-2,z,22)] = d.f[gpu_idx_global4(x,1,z,22)];
    d.f[gpu_idx_global4(x,NY-2,z,23)] = d.f[gpu_idx_global4(x,1,z,23)];
    d.f[gpu_idx_global4(x,NY-2,z,26)] = d.f[gpu_idx_global4(x,1,z,26)];
    #endif // D3Q27

    d.g[gpu_idx_global4(1,y,z,1)] = d.g[gpu_idx_global4(NX-2,y,z,1)];
    d.g[gpu_idx_global4(NX-2,y,z,2)] = d.g[gpu_idx_global4(1,y,z,2)];

    d.g[gpu_idx_global4(x,1,z,3)] = d.g[gpu_idx_global4(x,NY-2,z,3)];
    d.g[gpu_idx_global4(x,NY-2,z,4)] = d.g[gpu_idx_global4(x,1,z,4)];

    // ghost cell periodicity
    d.phi[gpu_idx_global3(0,y,z)] = d.phi[gpu_idx_global3(NX-2,y,z)];
    d.phi[gpu_idx_global3(NX-1,y,z)] = d.phi[gpu_idx_global3(1,y,z)];
    d.phi[gpu_idx_global3(x,0,z)] = d.phi[gpu_idx_global3(x,NY-2,z)];
    d.phi[gpu_idx_global3(x,NY-1,z)] = d.phi[gpu_idx_global3(x,1,z)];
}

#elif defined(DROPLET_CASE)

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool is_valid_edge = (x < NX && y < NY && z < NZ) &&
                            (x == 0 || x == NX-1 ||
                             y == 0 || y == NY-1 || 
                             z == 0 || z == NZ-1); 
    if (!is_valid_edge) return;                       

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.f[streamed_boundary_idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.g[streamed_boundary_idx4] = W_G[Q] * d.phi[idx3] - W_G[Q];
        }
    }
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    const idx_t top = gpu_idx_global3(x,y,z);
    const idx_t below = gpu_idx_global3(x,y,z-1);

    d.phi[top] = d.phi[below];
    d.rho[top] = d.rho[below];
    d.ux[top]  = d.ux[below];
    d.uy[top]  = d.uy[below];
    d.uz[top]  = d.uz[below];
}

#endif // FLOW_CASE

// ============================================================================================================== //

