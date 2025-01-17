#ifndef BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECT_HPP

#include <Expression.hpp>

#include "config.hpp"
#include "utilities/sharedDataTypes.hpp"

using boss::Span;

namespace adaptive {

std::string getSelectName(Select select);

template <typename T, typename U, typename F>
Span<int32_t> select(Select implementation, const Span<T>& column, U value, bool columnIsFirstArg,
                     F& predicate, Span<int32_t>&& candidateIndexes, size_t engineDOP = 1,
                     SelectOperatorState* state = nullptr);

} // namespace adaptive

#include "selectImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
