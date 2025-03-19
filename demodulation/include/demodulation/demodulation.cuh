#ifndef DEMODULATION
#define DEMODULATION

#include "cuda_runtime.h"
#include "filtering/filtering.cuh"

__global__ void cosMultiplierKernel(const float *inSignal, float *outSignal, const float signalAngleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        outSignal[tid] = inSignal[tid] * cos(signalAngleFreq*(float)tid);
}

__global__ void sinMultiplierKernel(const float *inSignal, float *outSignal, const float signalAngleFreq, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        outSignal[tid] = inSignal[tid] * sin(signalAngleFreq*(float)tid);
}

std::vector<float> demodulatorAM(const std::vector<float>& modulatedSignal, const float modulationFreq, const int sampleFreq){
    std::vector<float> temp(modulatedSignal.size(), 0.0); // промежуточный сигнал - умноженный на cos исходный
    float angleFreq = 2 * 3.1415 * modulationFreq / sampleFreq;

    int blockSize = 256;
    int gridSize = (modulatedSignal.size() + blockSize - 1) / blockSize;

    std::vector<float> demodulatedSignal(modulatedSignal.size(), 0.0);

    float *modulatedSignalPtr, *demodulatedSignalPtr, *tempPtr;
    cudaMalloc(&modulatedSignalPtr, modulatedSignal.size() * sizeof(float));
    cudaMalloc(&demodulatedSignalPtr, demodulatedSignal.size() * sizeof(float));
    cudaMalloc(&tempPtr, temp.size() * sizeof(float));

    // получение промежуточного сигнала
    cudaMemcpy(modulatedSignalPtr, modulatedSignal.data(), modulatedSignal.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "starting kernel" << std::endl;

    cosMultiplierKernel<<<gridSize, blockSize>>>(modulatedSignalPtr, tempPtr, angleFreq, modulatedSignal.size());

    cudaMemcpy(temp.data(), tempPtr, temp.size() * sizeof(float), cudaMemcpyDeviceToHost); // выгрузили промежуточный сигнал

    std::cout << "ended kernel and copied memo" << std::endl;

    // фильтрация промежуточного сигнала, убираем удвоенные частоты

    std::string filterCoefsFilePath = "C://Users//theiz//OneDrive//Desktop//EngPathStuff//cppSTC//GPUcourse//CUDAProjectCL//filters//demodAMforWAV.txt";
    FilterCoefficientsParser filterCoefsParser(filterCoefsFilePath);
    filterCoefsParser.readFilterCoefficients();

    std::cout << "parsed filter coefs" << std::endl;

    std::vector<float> numerator = filterCoefsParser.getNumerator();
    std::vector<float> denominator = filterCoefsParser.getDenominator();

    std::cout << numerator.size() << " " << denominator.size() << std::endl;

    filter(temp, demodulatedSignal, numerator, denominator);

    std::cout << "applied filter" << std::endl;

    return demodulatedSignal;
}

//std::vector<float> demodulatorFM(const std::vector<float>& modulatedSignal, std::vector<float>& demodulatedSignal, const float angleFreq, int n){
//    std::vector<float> temp(n, 0.0); // промежуточный сигнал - умноженный на cos исходный
//    int blockSize = 256;
//    int gridSize = (n + blockSize - 1) / blockSize;
//
//
//}

#endif