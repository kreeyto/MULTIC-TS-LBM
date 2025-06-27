#include "kernels.cuh"

__constant__ float W[FLINKS];
__constant__ float W_G[GLINKS];

__constant__ ci_t CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
    __constant__ float DATAZ[200];
#endif

LBMFields lbm;
DerivedFields dfields;
                                         
// =============================================================================================================================================================== //

void initDeviceVars() {
    size_t SIZE =        NX * NY * NZ          * sizeof(float);            
    size_t F_DIST_SIZE = NX * NY * NZ * FLINKS * sizeof(dtype_t); 
    size_t G_DIST_SIZE = NX * NY * NZ * GLINKS * sizeof(float); 

    checkCudaErrors(cudaMalloc(&lbm.rho,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ux,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uy,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uz,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pzz,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxz,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyz,   SIZE));

    checkCudaErrors(cudaMalloc(&lbm.phi,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ind,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normx, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffz,   SIZE));

    checkCudaErrors(cudaMalloc(&lbm.f,     F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&lbm.g,     G_DIST_SIZE));

    checkCudaErrors(cudaMalloc(&dfields.vorticity_mag, SIZE));
    checkCudaErrors(cudaMalloc(&dfields.q_criterion,   SIZE));

    checkCudaErrors(cudaMemset(lbm.ux,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uy,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uz,    0, SIZE));

    checkCudaErrors(cudaMemset(lbm.phi,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffx,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffy,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffz,   0, SIZE));

    checkCudaErrors(cudaMemcpyToSymbol(W,   &H_W,   FLINKS * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(W_G, &H_W_G, GLINKS * sizeof(float)));

    checkCudaErrors(cudaMemcpyToSymbol(CIX,   &H_CIX,   FLINKS * sizeof(ci_t)));
    checkCudaErrors(cudaMemcpyToSymbol(CIY,   &H_CIY,   FLINKS * sizeof(ci_t)));
    checkCudaErrors(cudaMemcpyToSymbol(CIZ,   &H_CIZ,   FLINKS * sizeof(ci_t)));

    #ifdef PERTURBATION
        checkCudaErrors(cudaMemcpyToSymbol(DATAZ, &H_DATAZ, 200 * sizeof(float)));
    #endif

    getLastCudaError("initDeviceVars: post-initialization");
}

