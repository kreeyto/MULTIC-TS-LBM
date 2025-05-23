#pragma once
#include "common.cuh"

extern __constant__ float CSSQ;
extern __constant__ float OMEGA;
extern __constant__ float GAMMA;
extern __constant__ float SIGMA;
extern __constant__ float COEFF_HE;

extern __constant__ float W[FLINKS];
extern __constant__ float W_G[GLINKS];

extern __constant__ int CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
    extern __constant__ float DATAZ[200];
#endif
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *normx, *normy, *normz, *ind;
    float *ffx, *ffy, *ffz;
    float *f, *g; 
};
 
/*
struct DerivedFields {
    float* ke;         // kinetic energy (local)
    float* tke;        // turbulent kinetic energy (com média temporal)
    float* qcriterion;
}
*/

extern LBMFields lbm;

void initDeviceVars();
