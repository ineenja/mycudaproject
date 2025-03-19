#include <vector>
#include <iostream>

#include "modulation/modulation.cuh"
#include "demodulation/demodulation.cuh"
#include "myparser/myparser.cuh"
//#include "samplefreqchange/samplefreqchange.cuh"

int main() {

    std::vector<float> testSignalIn = {0,1,1,0,0,1,1,0,0,1,1,0};

    float modulationFreq = 2000000;
    int sampleFreq = 44100;

    std::vector<float> modulatedSignal = modulatorAM(testSignalIn, modulationFreq, sampleFreq);

    for (float value : modulatedSignal){
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::vector<float> demodulatedSignal = demodulatorAM(modulatedSignal, modulationFreq, sampleFreq);

    for (float value : demodulatedSignal){
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}