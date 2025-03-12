#ifndef FILTERING
#define FILTERING

#include "cuda_runtime.h"
#include <iostream>

__global__ void memoryRecalculationKernel(float* memory, float* memoryTemp, const float *numerator, const float *denumerator,
                                          const float inputSignalSample, const float outputSignalSample, const unsigned int order){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < order - 1){
        memory[tid] = memoryTemp[tid + 1] + numerator[tid + 1] * inputSignalSample - denumerator[tid + 1] * outputSignalSample;
        printf("Thread %d: delayElements[%d] = %f\n", tid, tid, memory[tid]);
    } 
}

/// функция фильтрации сигнала, выполняется на CPU
void filter(std::vector<float>& inputSignal, std::vector<float>& outputSignal,
            std::vector<float>& numerator, std::vector<float>& denumerator){
    unsigned int order = numerator.size() > denumerator.size() ? numerator.size() : denumerator.size();

    std::vector<float> memory(order - 1, 0.0);
    std::vector<float> memoryTemp(order - 1, 0.0);

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
//        std::cout << outputSignal[k] << std::endl;

        cudaMemcpy(memoryPtr, memory.data(), memory.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(memoryTempPtr, memoryTemp.data(), memoryTemp.size() * sizeof(float), cudaMemcpyHostToDevice);

        memoryRecalculationKernel<<<gridSize, blockSize>>>(memoryPtr, memoryTempPtr, numeratorPtr, denumeratorPtr,
                                                           inputSignal[k], outputSignal[k], order);

        cudaMemcpy(memory.data(), memoryPtr, order * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(memoryTemp.data(), memoryPtr, order * sizeof(float), cudaMemcpyDeviceToHost);
//        std::cout << inputSignal[k] << " " << outputSignal[k] << " " << memory[0] << " " << memory[1] << std::endl;
//        std::cout << memoryTemp[0] << " " << memoryTemp[1] << std::endl;
//        std::cout << std::endl;
    }

    cudaFree(numeratorPtr);
    cudaFree(denumeratorPtr);
    cudaFree(memoryPtr);
    cudaFree(memoryTempPtr);
}





#endif