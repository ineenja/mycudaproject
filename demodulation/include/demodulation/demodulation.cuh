#ifndef DEMODULATION
#define DEMODULATION

#include "cuda_runtime.h"
#include "filtering/filtering.cuh"

__global__ void cosMultiplierKernel(const float *inSignal, float *outSignal, const float signalAngleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        outSignal[tid] = inSignal[tid] * cos(signalAngleFreq*(float)tid);
}

void demodulatorAM(const std::vector<float>& modulatedSignal, std::vector<float>& demodulatedSignal, const float angleFreq, int n){
    std::vector<float> temp(n, 0.0); // промежуточный сигнал - умноженный на cos исходный
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    float *modulatedSignalPtr, *demodulatedSignalPtr, *tempPtr;
    cudaMalloc(&modulatedSignalPtr, modulatedSignal.size() * sizeof(float));
    cudaMalloc(&demodulatedSignalPtr, demodulatedSignal.size() * sizeof(float));
    cudaMalloc(&tempPtr, temp.size() * sizeof(float));

    // получение промежуточного сигнала
    cudaMemcpy(modulatedSignalPtr, modulatedSignal.data(), modulatedSignal.size() * sizeof(float), cudaMemcpyHostToDevice);

    cosMultiplierKernel<<<gridSize, blockSize>>>(modulatedSignalPtr, tempPtr, angleFreq, n);

    cudaMemcpy(temp.data(), tempPtr, temp.size() * sizeof(float), cudaMemcpyDeviceToHost); // выгрузили промежуточный сигнал

    // фильтрация промежуточного сигнала, убираем удвоенные частоты

    std::vector<float> numerator = {0.9, 0.1, -0.2}; // нерекурсивная часть, кэфы bi, фильтра убирающего удвоенные частоты
    std::vector<float> denumerator = {1, 0.5, 0.01}; // рекурсивная часть, кэфы ai, фильтра убирающего удвоенные частоты

    filter(temp, demodulatedSignal, numerator, denumerator);
}

#endif