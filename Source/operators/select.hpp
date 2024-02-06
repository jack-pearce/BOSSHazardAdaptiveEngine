#ifndef BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECT_HPP

#include <Expression.hpp>

#include "utilities/sharedDataTypes.hpp"

using boss::Span;

namespace adaptive {

enum Select { Branch, Predication, Adaptive, AdaptiveParallel };

std::string getSelectName(Select select);

template <typename T, typename F>
Span<int32_t> select(Select implementation, const Span<T>& column, T value, bool columnIsFirstArg,
                     F& predicate, Span<int32_t>&& candidateIndexes, size_t dop = 1,
                     SelectOperatorState* state = nullptr, bool calibrationRun = false);

} // namespace adaptive

#include "selectImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
