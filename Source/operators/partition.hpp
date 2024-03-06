#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP

#include "HazardAdaptiveEngine.hpp"
#include "config.hpp"

#include <Expression.hpp>
#include <string>
#include <vector>

using boss::Span;

namespace adaptive {

std::string getPartitionName(PartitionOperators partitionImplementation);

template <typename T>
std::vector<int> partition(PartitionOperators partitionImplementation, int n, T* keys,
                           int radixBits = -1);

struct PartitionedJoinArguments {
  std::vector<ExpressionSpanArguments> tableOnePartitionsOfKeySpans;
  std::vector<ExpressionSpanArguments> tableOnePartitionsOfIndexSpans;
  std::vector<ExpressionSpanArguments> tableTwoPartitionsOfKeySpans;
  std::vector<ExpressionSpanArguments> tableTwoPartitionsOfIndexSpans;
};

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           const ExpressionSpanArguments& tableOneKeys,
                                           const ExpressionSpanArguments& tableTwoKeys,
                                           int dop = 1);

} // namespace adaptive

#include "partitionImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITION_HPP
