#ifndef BOSSHAZARDADAPTIVEENGINE_GROUP_H
#define BOSSHAZARDADAPTIVEENGINE_GROUP_H

#include "HazardAdaptiveEngine.hpp"
#include "config.hpp"

#include <Expression.hpp>
#include <optional>
#include <string>
#include <vector>

using boss::Span;

namespace adaptive {

enum Aggregation { Min, Max, Sum, Count };

template <typename T> using Aggregator = std::function<T(const T, const T, bool)>;

std::string getGroupName(Group groupImplementation);

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
group(Group implementation, int numKeys, ExpressionSpanArguments&& keySpans1,
      ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
      Aggregator<As>... aggregators);

} // namespace adaptive

#include "groupImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_GROUP_H
