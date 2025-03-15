#include <vector>
#include <iostream>

//#include "modulation/modulation.cuh"
//#include "demodulation/demodulation.cuh"
//#include "myparser/myparser.cuh"
#include "filtering/filtering.cuh"

int main() {

    std::vector<float> inputSignal = {1, 0, 0 ,0 ,0, 0, 0, 0, 0, 0};
    std::vector<float> outputSignal(inputSignal.size());
    std::vector<float> numerator = {0.9, 0.1, -0.2}; // нерекурсивная часть, кэфы bi
    std::vector<float> denumerator = {1, 0.5, 0.01}; // рекурсивная часть, кэфы ai

    filter(inputSignal, outputSignal, numerator, denumerator);

    std::cout << std::endl;
    for (float i : outputSignal){
        std::cout << i << " ";
    }

    return 0;
}