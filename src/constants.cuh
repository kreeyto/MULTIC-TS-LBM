#pragma once

#include "../include/utils_cuda.cuh"
#include "../include/velocity_sets.cuh"
#include "../include/perturbation_data.cuh"

#define JET_CASE
//#define DROPLET_CASE

#define RUN_MODE
//#define SAMPLE_MODE
//#define DEBUG_MODE

#ifdef RUN_MODE
    constexpr int MACRO_SAVE = 100;
    constexpr int NSTEPS = 30000;
#elif defined(SAMPLE_MODE)
    constexpr int MACRO_SAVE = 100;
    constexpr int NSTEPS = 1000;
#elif defined(DEBUG_MODE)
    constexpr int MACRO_SAVE = 1;
    constexpr int NSTEPS = 0;
#endif

#ifdef JET_CASE
    // domain size
    constexpr int MESH = 200;
    constexpr int DIAM = 20;
    constexpr int NX   = MESH;
    constexpr int NY   = MESH;
    constexpr int NZ   = MESH*2;
    // jet velocity
    constexpr float U_JET = 0.05; 
    // adimensional parameters
    constexpr int REYNOLDS = 5000; 
    constexpr int WEBER    = 500; 
    // general model parameters
    constexpr float VISC     = (U_JET * DIAM) / REYNOLDS;      // kinematic viscosity
    constexpr float TAU      = 0.5f + 3.0f * VISC;             // relaxation time
    constexpr float GAMMA    = 0.3f * 3.0f;                    // sharpening of the interface
    constexpr float SIGMA    = (U_JET * U_JET * DIAM) / WEBER; // surface tension coefficient
#elif defined(DROPLET_CASE)
    // domain size
    constexpr int MESH = 64;
    constexpr int RADIUS = 9; 
    constexpr int NX   = MESH;
    constexpr int NY   = MESH;
    constexpr int NZ   = MESH;
    // not used in this case, defined to avoid compilation errors
    constexpr float U_JET = 0.05; 
    // general model parameters
    constexpr float TAU      = 0.55f;       // relaxation time
    constexpr float GAMMA    = 0.15f * 5.0f; // sharpening of the interface
    constexpr float SIGMA    = 0.1f;          // surface tension coefficient
#endif // FLOW_CASE

// sponge parameters
constexpr float K    = 50.0f; // gain factor 
constexpr float P    = 3.0f;    // transition degree (polynomial)
constexpr int CELLS  = int(NZ/12);      // width    

// general model parameters and auxiliary constants
constexpr float CSSQ      = 1.0f / 3.0f;  // square of speed of sound
constexpr float OMEGA     = 1.0f / TAU;   // relaxation frequency
constexpr float OOS       = 1.0f / 6.0f;  // one over six
constexpr float OMCO      = 1.0f - OMEGA; // complementary of omega
constexpr float CSCO      = 1.0f - CSSQ;  // complementary of cssq

// sponge related auxiliary constants
constexpr float SPONGE    = float(CELLS) / float(NZ-1);                 // sponge width in normalized coordinates
constexpr float Z_START   = float(NZ-1-CELLS) / float(NZ-1);            // z coordinate where the sponge starts
constexpr float OMEGA_MAX = 1.0f / ((VISC * (K + 1.0f)) / CSSQ + 0.5f); // omega at z=max
constexpr float OMCO_MAX  = 1.0f - OMEGA_MAX;                           // complementary of omega at z=max