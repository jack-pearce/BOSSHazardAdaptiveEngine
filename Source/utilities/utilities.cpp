#include <cmath>
#include <numeric>
#include <stdexcept>

#include "config.hpp"
#include "utilities.hpp"

inline bool isPowerOfTwo(uint32_t n) { return n > 0 && (n & (n - 1)) == 0; }

int convertToValidDopValue(int dop) {
  if(dop == adaptive::config::LOGICAL_CORE_COUNT || isPowerOfTwo(static_cast<uint32_t>(dop))) {
    return dop;
  }
  return static_cast<int>(roundDownToPowerOf2(static_cast<uint32_t>(dop)));
}

uint32_t roundDownToPowerOf2(uint32_t num) {
  if(num <= 2) {
    return 1;
  }
  uint32_t msbPos = sizeof(uint32_t) * 8 - __builtin_clz(num) - 1; // NOLINT
  if((num & (num - 1)) == 0) {
    return 1U << (msbPos - 1);
  }
  return 1U << msbPos;
}

void setCardinalityEnvironmentVariable(int cardinality) {
  std::string value = std::to_string(cardinality);
  setenv("GROUP_RESULT_CARDINALITY", value.c_str(), 1);
}

double linearRegressionSlope(const std::vector<int>& x, const std::vector<long_long>& y) {
  size_t n = x.size();

  if(n != y.size() || n == 0) {
    throw std::invalid_argument("Vectors x and y must have the same non-zero size.");
  }

  double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);

  double numerator = 0.0;
  double denominator = 0.0;
  for(size_t i = 0; i < n; ++i) {
    numerator += (x[i] - mean_x) * (static_cast<double>(y[i]) - mean_y);
    denominator += std::pow(x[i] - mean_x, 2);
  }
  return numerator / denominator;
}
