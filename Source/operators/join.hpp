#ifndef BOSSHAZARDADAPTIVEENGINE_JOIN_HPP
#define BOSSHAZARDADAPTIVEENGINE_JOIN_HPP

#include "HazardAdaptiveEngine.hpp"
#include <Expression.hpp>

using boss::Span;

namespace adaptive {

struct JoinResultIndexes {
  ExpressionSpanArguments tableOneIndexes;
  ExpressionSpanArguments tableTwoIndexes;
};

template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2, const Span<int64_t>& indexes1,
                       const Span<int64_t>& indexes2);

template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2,
                       int dop);

template <typename T1, typename T2, typename T3, typename T4>
JoinResultIndexes join(const ExpressionSpanArguments& keySpansFirstKey1,
                       const ExpressionSpanArguments& keySpansSecondKey1,
                       const ExpressionSpanArguments& keySpansFirstKey2,
                       const ExpressionSpanArguments& keySpansSecondKey2,
                       int dop);

} // namespace adaptive

#include "joinImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_JOIN_HPP
