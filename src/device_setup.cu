#include "kernels.cuh"

__global__ void gpuInitFieldsAndDistributions(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;
    const int idx3 = gpuIdxGlobal3(x,y,z);

    // no implicit initialization even though rho=1 and phi=0.
    // just going for safety here, as f and g could be simplified.
    d.rho[idx3] = 1.0f;
    float rho_val = d.rho[idx3];
    float phi_val = d.phi[idx3];
    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        d.f[idx4] = W[Q] * rho_val;
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        d.g[idx4] = W_G[Q] * phi_val;
    }
}


__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float GAMMA;
__constant__ float INT_W;
__constant__ float SIGMA;
__constant__ float COEFF_HE;

__constant__ float W[FLINKS];
__constant__ float W_G[GLINKS];

__constant__ int CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
    __constant__ float DATAZ[200];
#endif

LBMFields lbm;
                                         
// =============================================================================================================================================================== //

void initDeviceVars() {
    size_t SIZE =        NX * NY * NZ          * sizeof(float);            
    size_t F_DIST_SIZE = NX * NY * NZ * FLINKS * sizeof(float); 
    size_t G_DIST_SIZE = NX * NY * NZ * GLINKS * sizeof(float); 

    checkCudaErrors(cudaMalloc(&lbm.phi,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.rho,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ux,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uy,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uz,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normx, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ind,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffz,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.f,     F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&lbm.g,     G_DIST_SIZE));

    checkCudaErrors(cudaMemset(lbm.phi,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ux,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uy,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uz,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.normx, 0, SIZE));
    checkCudaErrors(cudaMemset(lbm.normy, 0, SIZE));
    checkCudaErrors(cudaMemset(lbm.normz, 0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ind,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffx,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffy,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffz,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.f,     0, F_DIST_SIZE));
    checkCudaErrors(cudaMemset(lbm.g,     0, G_DIST_SIZE));

    checkCudaErrors(cudaMemcpyToSymbol(CSSQ,     &H_CSSQ,     sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(OMEGA,    &H_OMEGA,    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(INT_W,    &H_INT_W,    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(GAMMA,    &H_GAMMA,    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(SIGMA,    &H_SIGMA,    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(COEFF_HE, &H_COEFF_HE, sizeof(float)));

    checkCudaErrors(cudaMemcpyToSymbol(W,   &H_W,   FLINKS * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(W_G, &H_W_G, GLINKS * sizeof(float)));

    checkCudaErrors(cudaMemcpyToSymbol(CIX,   &H_CIX,   FLINKS * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(CIY,   &H_CIY,   FLINKS * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(CIZ,   &H_CIZ,   FLINKS * sizeof(int)));

    #ifdef PERTURBATION
        checkCudaErrors(cudaMemcpyToSymbol(DATAZ, &H_DATAZ, 200 * sizeof(float)));
    #endif

    getLastCudaError("initDeviceVars: post-initialization");
}

