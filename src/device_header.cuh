#pragma once
#include "constants.cuh"

extern __constant__ float W[FLINKS];
extern __constant__ float W_G[GLINKS];

extern __constant__ ci_t CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS], OPP[FLINKS];

#ifdef PERTURBATION
    extern __constant__ float DATAZ[200];
#endif
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *pxx, *pyy, *pzz, *pxy, *pxz, *pyz;
    float *ind, *normx, *normy, *normz;
    float *ffx, *ffy, *ffz;
    dtype_t *f; float *g; 
};

struct DerivedFields {
    float *vorticity_mag;
    float *q_criterion;
};

extern LBMFields lbm;
extern DerivedFields dfields;

void initDeviceVars();
