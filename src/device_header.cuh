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
    float *ind, *ffx, *ffy, *ffz;
    float *f, *g; 
};

extern LBMFields lbm;

void initDeviceVars();
