#include <gtest/gtest.h>
#include <iostream>
#include "harmparameters/harmparameters.h"


TEST(HarmParametersTest, CheckingParametersSet) {
    //HarmParameters(float Ampl, float Freq, float BeginningTime, float Duration, uint32_t SampleRate, uint32_t HarmID)
    float SampleAmpl = 1.0;
    float SampleFreq = 2.0;
    float SampleBTime = 3.0;
    float SampleDuration = 4.0;
    uint32_t SampleSampleRate = 5;
    uint32_t HarmID = 6;

    HarmParameters ToCheck = HarmParameters(SampleAmpl, SampleFreq, SampleBTime, SampleDuration, SampleSampleRate, HarmID);

    EXPECT_EQ(ToCheck.getHarmAmpl(), SampleAmpl);
    EXPECT_EQ(ToCheck.getHarmFreq(), SampleFreq);
    EXPECT_EQ(ToCheck.getBeginningTimeMS(), SampleBTime);
    EXPECT_EQ(ToCheck.getDurationMS(), SampleDuration);
    EXPECT_EQ(ToCheck.getSampleRate(), SampleSampleRate);
    EXPECT_EQ(ToCheck.getSignalID(), HarmID);
}

TEST(HarmParametersTest, CheckingParametersCalculated) {
    //HarmParameters(float Ampl, float Freq, float BeginningTime, float Duration, uint32_t SampleRate, uint32_t HarmID)
    float SampleAmpl = 1.0;
    float SampleFreq = 2.0;
    float SampleBTime = 3.0;
    float SampleDuration = 4.0;
    uint32_t SampleSampleRate = 50001;
    uint32_t HarmID = 6;

    HarmParameters ToCheck = HarmParameters(SampleAmpl, SampleFreq, SampleBTime, SampleDuration, SampleSampleRate, HarmID);

    uint32_t SampleSigType = 1;
    EXPECT_EQ(ToCheck.getSignalType(), SampleSigType);

    uint32_t SampleSignalLengthSamples = 200;
    ASSERT_EQ(ToCheck.getSignalLengthSamples(), SampleSignalLengthSamples);

    uint32_t SampleBeginningSampleN = 150;
    EXPECT_EQ(ToCheck.getBeginningSampleN(), SampleBeginningSampleN);
}

