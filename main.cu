#include <vector>
#include <iostream>

//#include "modulation/modulation.cuh"
//#include "demodulation/demodulation.cuh"
//#include "myparser/myparser.cuh"
#include "filtering/filtering.cuh"

int main() {

    int m = 5, n = 5;
    std::vector<float> inputSignal = {1, 0, 0 ,0 ,0};
    std::vector<float> outputSignal(inputSignal.size());
    std::vector<float> numerator = {2, 2, 3 ,1 ,2};
    std::vector<float> denumerator = {1, -1, 0.1, 2, 0.1};
    std::vector<float> memory(inputSignal.size(), 0.0);
    int order = 5;

    float *inputSignalPtr, *outputSignalPtr;
    float *numeratorPtr, *denumeratorPtr;
    float *memoryPtr;

    cudaMalloc(&inputSignalPtr, order * sizeof(float ));
    cudaMalloc(&outputSignalPtr, order * sizeof(float ));
    cudaMalloc(&numeratorPtr, n * sizeof(float ));
    cudaMalloc(&denumeratorPtr, m * sizeof(float ));
    cudaMalloc(&memoryPtr, order * sizeof(float ));

    cudaMemcpy(inputSignalPtr, inputSignal.data(), order * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outputSignalPtr, outputSignal.data(), order * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(numeratorPtr, numerator.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(denumeratorPtr, denumerator.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(memoryPtr, memory.data(), order * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (order + blockSize - 1) / blockSize;
    filterKernel<<<gridSize, blockSize>>>(inputSignalPtr, outputSignalPtr, order, numeratorPtr, denumeratorPtr, memoryPtr, order);

    cudaMemcpy(outputSignal.data(), outputSignalPtr, order * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inputSignalPtr);
    cudaFree(outputSignalPtr);
    cudaFree(numeratorPtr);
    cudaFree(denumeratorPtr);
    cudaFree(memoryPtr);

    for (float i : outputSignal){
        std::cout << i << " ";
    }

    return 0;
}