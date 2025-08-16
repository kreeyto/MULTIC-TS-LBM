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
    const float uz_in = 
    #ifdef PERTURBATION
        /* apply perturbation */ U_JET * (1.0f + DATAZ[(STEP/MACRO_SAVE)%200] * 10.0f);
    #else
        /* straightforward */ U_JET;
    #endif 

    d.uz[idx3_in] = uz_in;

    int neighbor_idx = gpu_idx_global3(x,y,z+1);
    float feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,5);
    float fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                                d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                                d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],5);
    d.f[gpu_idx_global4(x,y,z+1,5)] = to_dtype(feq + OMCO * fneq_reg);

    feq = gpu_compute_truncated_equilibria(1.0f,0.0f,0.0f,uz_in,5);
    d.g[gpu_idx_global4(x,y,z+1,5)] = feq;

    neighbor_idx = gpu_idx_global3(x+1,y,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,9);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],9);
    d.f[gpu_idx_global4(x+1,y,z+1,9)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x,y+1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,11);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],11);
    d.f[gpu_idx_global4(x,y+1,z+1,11)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x-1,y,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,16);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],16);
    d.f[gpu_idx_global4(x-1,y,z+1,16)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x,y-1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,18);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],18);
    d.f[gpu_idx_global4(x,y-1,z+1,18)] = to_dtype(feq + OMCO * fneq_reg);

    #ifdef D3Q27
    neighbor_idx = gpu_idx_global3(x+1,y+1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,19);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],19);
    d.f[gpu_idx_global4(x+1,y+1,z+1,19)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x-1,y-1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,22);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],22);
    d.f[gpu_idx_global4(x-1,y-1,z+1,22)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x+1,y-1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,23);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],23);
    d.f[gpu_idx_global4(x+1,y-1,z+1,23)] = to_dtype(feq + OMCO * fneq_reg);

    neighbor_idx = gpu_idx_global3(x-1,y+1,z+1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],0.0f,0.0f,uz_in,25);
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
    
    //d.rho[idx_outer] = d.rho[idx_inner];
    d.phi[idx_outer] = d.phi[idx_inner];
    d.ux[idx_outer] = d.ux[idx_inner];
    d.uy[idx_outer] = d.uy[idx_inner];
    d.uz[idx_outer] = d.uz[idx_inner];

    const float ux_out = d.ux[idx_outer];
    const float uy_out = d.uy[idx_outer];
    const float uz_out = d.uz[idx_outer];

    int neighbor_idx = gpu_idx_global3(x,y,z-1);
    float feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,6);
    float fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                                d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                                d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],6);
    d.f[gpu_idx_global4(x,y,z-1,6)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    feq = gpu_compute_truncated_equilibria(d.phi[neighbor_idx],ux_out,uy_out,uz_out,6);
    d.g[gpu_idx_global4(x,y,z-1,6)] = feq;

    neighbor_idx = gpu_idx_global3(x-1,y,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,10);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],10);
    d.f[gpu_idx_global4(x-1,y,z-1,10)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,12);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],12);
    d.f[gpu_idx_global4(x,y-1,z-1,12)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x+1,y,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,15);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],15);
    d.f[gpu_idx_global4(x+1,y,z-1,15)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,17);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],17);
    d.f[gpu_idx_global4(x,y+1,z-1,17)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    #ifdef D3Q27
    neighbor_idx = gpu_idx_global3(x-1,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,20);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],20);
    d.f[gpu_idx_global4(x-1,y-1,z-1,20)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x+1,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,21);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],21);
    d.f[gpu_idx_global4(x+1,y+1,z-1,21)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x-1,y+1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,24);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],24);
    d.f[gpu_idx_global4(x-1,y+1,z-1,24)] = to_dtype(feq + OMCO_MAX * fneq_reg);

    neighbor_idx = gpu_idx_global3(x+1,y-1,z-1);
    feq = gpu_compute_equilibria(d.rho[neighbor_idx],ux_out,uy_out,uz_out,26);
    fneq_reg = gpu_compute_non_equilibria(d.pxx[neighbor_idx],d.pyy[neighbor_idx],d.pzz[neighbor_idx],
                                          d.pxy[neighbor_idx],d.pxz[neighbor_idx],d.pyz[neighbor_idx],
                                          d.ux[neighbor_idx],d.uy[neighbor_idx],d.uz[neighbor_idx],26);
    d.f[gpu_idx_global4(x+1,y-1,z-1,26)] = to_dtype(feq + OMCO_MAX * fneq_reg);
    #endif // D3Q27
}

__global__ void gpuApplyPeriodicX(LBMFields d) {
    const int y = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if (y <= 0 || y >= NY-1 || z <= 0 || z >= NZ-1) return;
    if (y >= NY || z >= NZ || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t bL = gpu_idx_global3(1,y,z);
    const idx_t bR = gpu_idx_global3(NX-2,y,z);

    // positive x contributions
    copy_dirs<dtype_t,1,7,9,13,15>(d.f,bL,bR);   
    #ifdef D3Q27
    copy_dirs<dtype_t,19,21,23,26>(d.f,bL,bR);
    #endif // D3Q27

    // negative x contributions
    copy_dirs<dtype_t,2,8,10,14,16>(d.f,bR,bL); 
    #ifdef D3Q27
    copy_dirs<dtype_t,20,22,24,25>(d.f,bR,bL);
    #endif // D3Q27

    d.g[1*PLANE+bL] = d.g[1*PLANE+bR];
    d.g[2*PLANE+bR] = d.g[2*PLANE+bL];
    d.phi[gpu_idx_global3(0,y,z)] = d.phi[bR];
    d.phi[gpu_idx_global3(NX-1,y,z)] = d.phi[bL];
}

__global__ void gpuApplyPeriodicY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if (x <= 0 || x >= NX-1 || z <= 0 || z >= NZ-1) return;
    if (x >= NX || z >= NZ || 
        x == 0 || x == NX-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t bB = gpu_idx_global3(x,1,z);
    const idx_t bT = gpu_idx_global3(x,NY-2,z);

    // positive y contributions
    copy_dirs<dtype_t,3,7,11,14,17>(d.f,bB,bT);
    #ifdef D3Q27
    copy_dirs<dtype_t,19,21,24,25>(d.f,bB,bT);
    #endif // D3Q27

    // negative y contributions
    copy_dirs<dtype_t,4,8,12,13,18>(d.f,bT,bB);
    #ifdef D3Q27
    copy_dirs<dtype_t,20,22,23,26>(d.f,bT,bB);
    #endif // D3Q27

    d.g[3*PLANE+bB] = d.g[3*PLANE+bT];
    d.g[4*PLANE+bT] = d.g[4*PLANE+bB];
    d.phi[gpu_idx_global3(x,0,z)] = d.phi[bT];
    d.phi[gpu_idx_global3(x,NY-1,z)] = d.phi[bB];
}

#elif defined(DROPLET_CASE)

// still undefined

#endif // FLOW_CASE

// ============================================================================================================== //

