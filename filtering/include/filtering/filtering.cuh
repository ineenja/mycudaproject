#ifndef FILTERING
#define FILTERING

#include "cuda_runtime.h"
#include <iostream>

__global__ void memoryRecalculationKernel(float* memory, float* memoryTemp, const float *numerator, const float *denumerator,
                                          const float inputSignalSample, const float outputSignalSample, const unsigned int order){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid <= order - 2){
        memory[tid] = memoryTemp[tid + 1] + numerator[tid + 1] * inputSignalSample - denumerator[tid + 1] * outputSignalSample;
    } else {
        if (tid == order - 1){
            memory[tid] = numerator[tid + 1] * inputSignalSample - denumerator[tid + 1] * outputSignalSample;
        }
    }
}

/// функция фильтрации сигнала, выполняется на CPU
void filter(std::vector<float>& inputSignal, std::vector<float>& outputSignal,
            std::vector<float>& numerator, std::vector<float>& denumerator){
    unsigned int order = numerator.size() > denumerator.size() ? numerator.size() : denumerator.size();
    order--;

    std::vector<float> memory(order, 0.0);
    std::vector<float> memoryTemp(order, 0.0);

    float *numeratorPtr, *denumeratorPtr, *memoryPtr, *memoryTempPtr;
    cudaMalloc(&numeratorPtr, numerator.size() * sizeof(float));
    cudaMalloc(&denumeratorPtr, denumerator.size() * sizeof(float));
    cudaMalloc(&memoryPtr, memory.size() * sizeof(float));
    cudaMalloc(&memoryTempPtr, memoryTemp.size() * sizeof(float));

    cudaMemcpy(numeratorPtr, numerator.data(), numerator.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(denumeratorPtr, denumerator.data(), denumerator.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (order + blockSize - 1) / blockSize;

    for (unsigned int k = 0; k < inputSignal.size(); ++k){
        outputSignal[k] = (inputSignal[k] * numerator[0] + memory[0]) * denumerator[0];

        cudaMemcpy(memoryPtr, memory.data(), memory.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(memoryTempPtr, memoryTemp.data(), memoryTemp.size() * sizeof(float), cudaMemcpyHostToDevice);

        memoryRecalculationKernel<<<gridSize, blockSize>>>(memoryPtr, memoryTempPtr, numeratorPtr, denumeratorPtr,
                                                           inputSignal[k], outputSignal[k], order);

        cudaMemcpy(memory.data(), memoryPtr, memory.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(memoryTemp.data(), memoryPtr, memoryTemp.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(numeratorPtr);
    cudaFree(denumeratorPtr);
    cudaFree(memoryPtr);
    cudaFree(memoryTempPtr);
}





#endif