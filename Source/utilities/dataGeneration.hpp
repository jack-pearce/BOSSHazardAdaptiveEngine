#ifndef BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP

#include "HazardAdaptiveEngine.hpp"

namespace adaptive {

template <typename T> void copyVector(const std::vector<T>& source, std::vector<T>& destination);

template <typename T> std::vector<T> generateRandomisedUniqueValuesInMemory(size_t n);

template <typename T>
std::vector<T> generateUniformDistributionWithSetCardinality(int n, int upperBound, int cardinality,
                                                             int seed = 1);

template <typename T>
std::vector<T> generateUniformDistribution(size_t n, T lowerBound, T upperBound, int seed = 1);

template<typename T>
ExpressionSpanArguments loadVectorIntoSpans(std::vector<T> vector);

template<typename T>
ExpressionSpanArgument loadVectorIntoSpan(std::vector<T> vector);

template<typename T>
std::vector<boss::Span<T>> shallowCopySpan(ExpressionSpanArgument& span);

} // namespace adaptive

#include "dataGenerationImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP
