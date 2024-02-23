#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP

#include "HazardAdaptiveEngine.hpp"

#include <Expression.hpp>
#include <string>
#include <vector>

using boss::Span;

namespace adaptive {

enum PartitionOperators { RadixBitsFixed, RadixBitsAdaptive };

std::string getPartitionName(PartitionOperators partitionImplementation);

template <typename T>
std::vector<int> partition(PartitionOperators partitionImplementation, int n, T* keys,
                           int radixBits = -1);

struct PartitionedJoinArguments {
  ExpressionSpanArguments tableOneKeySpans;
  ExpressionSpanArguments tableOneIndexSpans;
  ExpressionSpanArguments tableTwoKeySpans;
  ExpressionSpanArguments tableTwoIndexSpans;
};

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           const ExpressionSpanArguments& tableOneKeys,
                                           const ExpressionSpanArguments& tableTwoKeys);

} // namespace adaptive

#include "partitionImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP
