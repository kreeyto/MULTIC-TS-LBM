#include "kernels.cuh"

__global__ void gpuDerivedFields(LBMFields lbm, DerivedFields dfields) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx = gpu_idx_global3(x, y, z);

    const float dudx = (lbm.ux[gpu_idx_global3(x+1,y,z)] - lbm.ux[gpu_idx_global3(x-1,y,z)]) * 0.5f;
    const float dudy = (lbm.ux[gpu_idx_global3(x,y+1,z)] - lbm.ux[gpu_idx_global3(x,y-1,z)]) * 0.5f;
    const float dudz = (lbm.ux[gpu_idx_global3(x,y,z+1)] - lbm.ux[gpu_idx_global3(x,y,z-1)]) * 0.5f;

    const float dvdx = (lbm.uy[gpu_idx_global3(x+1,y,z)] - lbm.uy[gpu_idx_global3(x-1,y,z)]) * 0.5f;
    const float dvdy = (lbm.uy[gpu_idx_global3(x,y+1,z)] - lbm.uy[gpu_idx_global3(x,y-1,z)]) * 0.5f;
    const float dvdz = (lbm.uy[gpu_idx_global3(x,y,z+1)] - lbm.uy[gpu_idx_global3(x,y,z-1)]) * 0.5f;

    const float dwdx = (lbm.uz[gpu_idx_global3(x+1,y,z)] - lbm.uz[gpu_idx_global3(x-1,y,z)]) * 0.5f;
    const float dwdy = (lbm.uz[gpu_idx_global3(x,y+1,z)] - lbm.uz[gpu_idx_global3(x,y-1,z)]) * 0.5f;
    const float dwdz = (lbm.uz[gpu_idx_global3(x,y,z+1)] - lbm.uz[gpu_idx_global3(x,y,z-1)]) * 0.5f;

    const float vort_x = dwdy - dvdz;
    const float vort_y = dudz - dwdx;
    const float vort_z = dvdx - dudy;

    const float vorticity_mag = sqrtf(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z);
    dfields.vorticity_mag[idx] = vorticity_mag;

    const float Sxx = dudx;
    const float Syy = dvdy;
    const float Szz = dwdz;

    const float Sxy = 0.5f * (dudy + dvdx);
    const float Sxz = 0.5f * (dudz + dwdx);
    const float Syz = 0.5f * (dvdz + dwdy);

    const float Omega_xy = 0.5f * (dudy - dvdx);
    const float Omega_xz = 0.5f * (dudz - dwdx);
    const float Omega_yz = 0.5f * (dvdz - dwdy);

    const float S_norm2 = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0f*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);

    const float Omega_norm2 = 2.0f*(Omega_xy*Omega_xy + Omega_xz*Omega_xz + Omega_yz*Omega_yz);

    const float Q = 0.5f * (Omega_norm2 - S_norm2);
    
    dfields.q_criterion[idx] = Q;
    // good q values float around 0.000005
}
