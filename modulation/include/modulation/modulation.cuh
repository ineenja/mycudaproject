#ifndef MODULATION
#define MODULATION

#include "cuda_runtime.h"

__global__ void modulatorAM(const float *infoSignal, float *modulatedSignal, const float angleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        modulatedSignal[tid] = infoSignal[tid] * cos(angleFreq*(float)tid);
}

__global__ void modulatorFM(const float *infoSignal, float *modulatedSignal, const float angleFreq, const float ampl, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        modulatedSignal[tid] = ampl * cos(angleFreq*(float)tid + infoSignal[tid]);
}

#endif