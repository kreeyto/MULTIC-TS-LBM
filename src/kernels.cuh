#pragma once
#include "device_header.cuh"
#include "device_functions.cuh"

// ======================================================================================================= //
//                                          INITIALIZATION                                                 //
// ======================================================================================================= //
__global__ void gpuInitFieldsAndDistributions(LBMFields d); // initialize fields and distributions         //
// ======================================================================================================= //
//                                      PHASE FIELD CALCULATIONS                                           //
// ======================================================================================================= //
__global__ void gpuPhi(LBMFields d);       // order variable summation                                     //
__global__ void gpuGradients(LBMFields d); // gradients, normals, curvature                                //
__global__ void gpuForces(LBMFields d);    // forces, curvature, phase field                               //
// ======================================================================================================= //
//                                       FLUID FIELD EVOLUTION                                             //
// ======================================================================================================= //
__global__ void gpuCollisionStream(LBMFields d);  // moments, fused collision + streaming                  //
__global__ void gpuEvolvePhaseField(LBMFields d); // advection-diffusion of the interface                  //
// ======================================================================================================= //
//                                        BOUNDARY CONDITIONS                                              //
// ======================================================================================================= //
__global__ void gpuApplyInflow(LBMFields d, const int STEP); // inflow at z=0                              //
__global__ void gpuReconstructBoundaries(LBMFields d);       // non-equilibrium extrapolation              //
__global__ void gpuApplyPeriodicXY(LBMFields d);             // periodicity in xy                          //
__global__ void gpuApplyOutflow(LBMFields d);                // neumann at z=nz-1                          //
// ======================================================================================================= //
//                                          DERIVED FIELDS                                                 //
// ======================================================================================================= //
__global__ void gpuDerivedFields(LBMFields lbm, DerivedFields dfields); // vorticity, Q-criterion, etc.    //
// ======================================================================================================= //

