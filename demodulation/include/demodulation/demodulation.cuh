#ifndef DEMODULATION
#define DEMODULATION

#include "cuda_runtime.h"

__global__ void demodulatorAM(const float *modulatedSignal, float *demodulatedSignal, const float angleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        demodulatedSignal[tid] = modulatedSignal[tid] / cos(angleFreq*(float)tid);
}

__global__ void demodulatorFM(const float *modulatedSignal, float *demodulatedSignal, const float angleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        demodulatedSignal[tid] = modulatedSignal[tid] / cos(angleFreq*(float)tid);
}

#endif