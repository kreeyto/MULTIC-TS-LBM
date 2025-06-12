#pragma once
#include "common.cuh"

extern __constant__ float W[FLINKS];
extern __constant__ float W_G[GLINKS];

extern __constant__ int CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
    extern __constant__ float DATAZ[200];
#endif
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *normx, *normy, *normz;
    float *ffx, *ffy, *ffz;
    dtype *f;
    float *g; 
};

extern LBMFields lbm;

void initDeviceVars();
