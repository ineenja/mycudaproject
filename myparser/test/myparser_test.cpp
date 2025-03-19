#include <gtest/gtest.h>
#include <iostream>
#include "myparser/myparser.cuh"


TEST(ParserTest, FilterCoefficientsReadingTest) {
    std::string filePath = "test.txt";

    FilterCoefficientsParser testCoefsParser(filePath);

    testCoefsParser.readFilterCoefficients();

    std::vector<float> numerator;
    std::vector<float> denominator;

    int test = 1;
    for (float value : numerator){
        EXPECT_EQ(value,test++);
    }
    for (float value : denominator){
        EXPECT_EQ(value,test++);
    }
}



