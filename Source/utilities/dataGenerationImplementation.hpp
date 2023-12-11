#ifndef BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP

#include <cassert>
#include <iostream>
#include <random>
#include <set>

namespace adaptive {

template <typename T> void copyVector(const std::vector<T>& source, std::vector<T>& destination) {
  if(destination.size() != source.size()) {
    destination.resize(source.size());
  }
  std::copy(source.begin(), source.end(), destination.begin());
}

template <typename T> std::vector<T> generateRandomisedUniqueValuesInMemory(size_t n) {
  // Fisher–Yates shuffle

  static_assert(std::is_integral<T>::value, "Must be an integer type");

  std::vector<T> data(n);

  for(T i = 1; i <= static_cast<T>(n); ++i) {
    data[i - 1] = i;
  }

  unsigned int seed = 1;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(1, 1);
  int j;

  for(int i = static_cast<int32_t>(n) - 1; i >= 0; --i) {
    dis = std::uniform_int_distribution<int>(0, i);
    j = dis(gen);
    std::swap(data[i], data[j]);
  }

  return data;
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP
