#pragma once
#include "device_header.cuh"
#include "device_functions.cuh"

// ======================================================================================================= //
//                                          INITIALIZATION                                                 //
// ======================================================================================================= //
__global__ void gpuInitDropletShape(LBMFields d);  // initialize the droplet shape for the droplet case    //
__global__ void gpuInitFields(LBMFields d);        // initialize fields                                    //
__global__ void gpuInitDistributions(LBMFields d); // initialize distributions                             //
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
__global__ void gpuApplyOutflow(LBMFields d);                // neumann at z=nz-1                          //
__global__ void gpuApplyPeriodic(LBMFields d);               // periodic at x-y                            //
// ======================================================================================================= //
//                                          DERIVED FIELDS                                                 //
// ======================================================================================================= //
__global__ void gpuDerivedFields(LBMFields lbm, DerivedFields dfields); // vorticity, Q-criterion, etc.    //
// ======================================================================================================= //

