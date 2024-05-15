#ifndef BOSSHAZARDADAPTIVEENGINE_UTILITIES_HPP
#define BOSSHAZARDADAPTIVEENGINE_UTILITIES_HPP

#include <iostream>
#include <papi.h>
#include <vector>

uint32_t roundDownToPowerOf2(uint32_t num);

void setCardinalityEnvironmentVariable(int cardinality);

double linearRegressionSlope(const std::vector<int>& x, const std::vector<long_long>& y);

#endif // BOSSHAZARDADAPTIVEENGINE_UTILITIES_HPP
