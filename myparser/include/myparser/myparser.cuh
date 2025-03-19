#ifndef MYPARSER
#define MYPARSER

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

class FilterCoefficientsParser{

public:

    explicit FilterCoefficientsParser(const std::string& filePath){
        std::fstream file;
        file.open(filePath, std::ios::in | std::ios::out);

        if (file.is_open()){

            std::string s;

            while(std::getline(file, s))
            {
                fileStrings.push_back(s);
            }
        }
        file.close();
    }

    void readFilterCoefficients(){
        for (size_t i = 0; i < fileStrings.size(); ++i){
            if (fileStrings[i] == "numerator:"){
                ++i;
                while(fileStrings[i] != "endofcoefs"){
                    numerator.push_back(std::stof(fileStrings[i]));
                    ++i;
                }
            }
            if (fileStrings[i] == "denominator:"){
                ++i;
                while(fileStrings[i] != "endofcoefs"){
                    denominator.push_back(std::stof(fileStrings[i]));
                    ++i;
                }
            }
        }
    }

    std::vector<float> getNumerator(){
        return numerator;
    };
    std::vector<float> getDenominator(){
        return denominator;
    }

private:

    std::vector<std::string> fileStrings;

    std::vector<float> numerator;
    std::vector<float> denominator;

};

#endif