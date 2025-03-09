#ifndef FILTERING
#define FILTERING

#include "cuda_runtime.h"
#include <iostream>

/// альтернативный вариант пересчета внутреннего состояния фильтра на каждой итерации
__device__ void innerStateChanging(float* memory, const float *numerator, const float *denumerator,
                                   const float inputSignalSample, const float outputSignalSample, const int order){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < order){
        memory[tid] = memory[tid + 1] + numerator[tid] * inputSignalSample - denumerator[tid] * outputSignalSample;
    }
}

/// реализация фильтра с транспонированной формой
__global__ void filterKernel(const float *inputSignal, float *outputSignal, const int inputSignalSize,
                       const float *numerator, const float *denumerator, float* memory, const int order) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < inputSignalSize) {
        outputSignal[tid] = inputSignal[tid] * numerator[0] + memory[0];
//        for (int k = 0; k < order-1; ++k){
//            memory[k] = memory[k + 1] + numerator[k] * inputSignal[tid] - denumerator[k] * outputSignal[tid];
//        }
        innerStateChanging(memory, numerator, denumerator, inputSignal[tid], outputSignal[tid], order);
    }
}



#endif