#ifndef SAMPLEFREQCHANGE
#define SAMPLEFREQCHANGE

#include "cuda_runtime.h"
#include "filtering/filtering.cuh"
#include <vector>
#include <iostream>

// Функция для нахождения наибольшего общего делителя (НОД) двух чисел
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Функция для нахождения наименьшего общего кратного (НОК) двух чисел
int lcm(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return (a * b) / gcd(a, b);
}

__global__ void inSignalZeroesAdditionKernel(float* signal, float* signalWithZeros, const int upFreqCoef, const size_t n){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid <= n){
        if (tid % upFreqCoef == 0){
            signalWithZeros[tid] = signal[tid];
        }
    }
}

__global__ void signalDecimationKernel(float* signal, float* signalDecimated, const int downFreqCoef, const size_t n){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid <= n){
        if (tid % downFreqCoef == 0){
            signalDecimated[tid] = signal[tid];
        }
    }
}

std::vector<float> changeSampleFrequency(const std::vector<float>& inSignal, const int oldSampleFreq, const int newSampleFreq){
    int LCMofFreqs = lcm(oldSampleFreq, newSampleFreq);
    int upFreqCoef = LCMofFreqs / oldSampleFreq;
    int downFreqCoef = LCMofFreqs / newSampleFreq;

    std::vector<float> tempSignal(inSignal.size()*upFreqCoef, 0.0);

    int blockSize = 256;
    int gridSize = (tempSignal.size() + blockSize - 1) / blockSize;
    float *inSignalPtr, *tempSignalPtr;

    // промежуточно увеличиваем частоту дискретизации, вставляем нули
    cudaMalloc(&inSignalPtr, inSignal.size() * sizeof(float));
    cudaMalloc(&tempSignalPtr, tempSignal.size() * sizeof(float));

    cudaMemcpy(inSignalPtr, inSignal.data(), inSignal.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tempSignalPtr, tempSignal.data(), tempSignal.size() * sizeof(float), cudaMemcpyHostToDevice);

    inSignalZeroesAdditionKernel<<<gridSize, blockSize>>>(inSignalPtr, tempSignalPtr, upFreqCoef, tempSignal.size());

    cudaMemcpy(tempSignal.data(), tempSignalPtr, tempSignal.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inSignalPtr);
    cudaFree(tempSignalPtr);
    // после вставления нулей фильтруем сигнал с ними для получения промежуточного сигнала

    // параметры фильтра
    std::vector<float> numerator = {0.9, 0.1, -0.2}; // нерекурсивная часть, кэфы bi
    for (int i = 0; i < numerator.size(); ++i){
        numerator[i] = numerator[i] * upFreqCoef;
    }
    std::vector<float> denumerator = {1, 0.5, 0.01}; // рекурсивная часть, кэфы ai

    std::vector<float> filteredSignal(tempSignal.size(), 0.0);

    filter(tempSignal, filteredSignal, numerator, denumerator);


    // децимация промежуточного сигнала
    gridSize = (filteredSignal.size() + blockSize - 1) / blockSize;
    float *filteredSignalPtr, *outSignalPtr;
    std::vector<float> outSignal(tempSignal.size() / downFreqCoef, 0.0);

    cudaMalloc(&filteredSignalPtr, filteredSignal.size() * sizeof(float));
    cudaMalloc(&outSignalPtr, outSignal.size() * sizeof(float));

    cudaMemcpy(filteredSignalPtr, filteredSignal.data(), filteredSignal.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outSignalPtr, outSignal.data(), outSignal.size() * sizeof(float), cudaMemcpyHostToDevice);

    signalDecimationKernel<<<gridSize, blockSize>>>(filteredSignalPtr, outSignalPtr, downFreqCoef, filteredSignal.size());

    cudaMemcpy(outSignal.data(), outSignalPtr, outSignal.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return outSignal;
}

#endif