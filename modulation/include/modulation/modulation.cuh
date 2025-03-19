#ifndef MODULATION
#define MODULATION

#include "cuda_runtime.h"
#include <vector>

__global__ void modulatorAMKernel(const float *infoSignal, float *modulatedSignal, const float angleFreq, size_t n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        modulatedSignal[tid] = infoSignal[tid] * cos(angleFreq*(float)tid);
}

__global__ void modulatorFMKernel(const float *infoSignal, float *modulatedSignal, const float angleFreq, const float ampl, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        modulatedSignal[tid] = ampl * cos(angleFreq*(float)tid + infoSignal[tid]);
}

std::vector<float> modulatorAM(const std::vector<float>& infoSignal, const float freq, const int sampleFreq){
    float angleFreq = 2 * 3.1415 * freq / sampleFreq;

    std::vector<float> modulatedSignal(infoSignal.size(),0.0);

    float *infoSignalPtr, *modulatedSignalPtr;

    cudaMalloc(&infoSignalPtr, infoSignal.size() * sizeof(float));
    cudaMalloc(&modulatedSignalPtr, modulatedSignal.size() * sizeof(float));

    cudaMemcpy(infoSignalPtr, infoSignal.data(), infoSignal.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(modulatedSignalPtr, modulatedSignal.data(), modulatedSignal.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (infoSignal.size() + blockSize - 1) / blockSize;

    modulatorAMKernel<<<gridSize, blockSize>>>(infoSignalPtr, modulatedSignalPtr, angleFreq, infoSignal.size());

    cudaMemcpy(modulatedSignal.data(), modulatedSignalPtr, modulatedSignal.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(infoSignalPtr);
    cudaFree(modulatedSignalPtr);

    return modulatedSignal;
}



#endif