#ifndef BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>

#include "HazardAdaptiveEngine.hpp"

namespace adaptive {

template <typename T> void copyVector(const std::vector<T>& source, std::vector<T>& destination) {
  if(destination.size() != source.size()) {
    destination.resize(source.size());
  }
  std::copy(source.begin(), source.end(), destination.begin());
}

template <typename T>
inline T scaleNumberLogarithmically(T number, int startingUpperBound, int targetUpperBound) {
  double scaledValue = log(number) / log(startingUpperBound);
  double scaledNumber = pow(targetUpperBound, scaledValue);
  return std::round(scaledNumber);
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

template <typename T>
std::vector<T> generateUniformDistribution(size_t n, T lowerBound, T upperBound, int seed) {
  static_assert(std::is_integral<T>::value, "Must be an integer type");

  std::mt19937 gen(seed);

  std::uniform_int_distribution<T> distribution(lowerBound, upperBound);

  std::vector<T> data;
  data.reserve(n);

  for(size_t i = 0; i < n; ++i) {
    data.push_back(distribution(gen));
  }
  return data;
}

template <typename T>
std::vector<T> generateUniformDistributionWithSetCardinality(int n, int upperBound, int cardinality,
                                                             int seed) {
  assert(n >= cardinality);

  if(cardinality == 1) {
    return std::vector<T>(n, upperBound);
  }

  int baselineDuplicates = n / cardinality;
  int remainingValues = n % cardinality;
  std::vector<T> data;
  data.reserve(n);

  for(int section = 0; section < cardinality; section++) {
    for(int elemInSection = 0; elemInSection < (baselineDuplicates + (section < remainingValues));
        elemInSection++) {
      data.push_back(section + 1);
    }
  }

  std::default_random_engine rng(seed);
  std::shuffle(data.begin(), data.end(), rng);

  if(upperBound != cardinality) {
    for(int i = 0; i < n; i++) {
      data[i] = scaleNumberLogarithmically(data[i], cardinality, upperBound);
    }
  }

  return data;
}

template<typename T>
ExpressionSpanArguments loadVectorIntoSpans(std::vector<T> vector) {
  ExpressionSpanArguments spans;
  spans.emplace_back(boss::Span<T>(std::vector(vector)));
  return spans;
}

template<typename T>
ExpressionSpanArgument loadVectorIntoSpan(std::vector<T> vector) {
  return boss::Span<T>(std::vector(vector));
}

template<typename T>
std::vector<boss::Span<T>> shallowCopySpan(ExpressionSpanArgument& untypedSpan) {
  std::vector<boss::Span<T>> spans;
  auto& typedSpan = get<boss::Span<T>>(untypedSpan);
  spans.emplace_back(boss::Span<T>(typedSpan.begin(), typedSpan.size(), [](){}));
  return spans;
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_DATAGENERATIONIMPLEMENTATION_HPP
