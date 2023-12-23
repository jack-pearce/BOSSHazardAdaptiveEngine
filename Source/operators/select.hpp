#ifndef BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECT_HPP

#include <Expression.hpp>

using boss::Span;

namespace adaptive {

enum SelectImplementation { Branch_, Predication_ };
enum Select { Branch, Predication, Adaptive, AdaptiveParallel};

std::string getSelectName(Select select);

template <typename T, typename F>
Span<uint32_t> select(Select implementation, const Span<T>& column, T value, bool columnIsFirstArg, F& predicate,
                      Span<uint32_t>&& candidateIndexes, size_t dop = 1, bool calibrationRun = false);

} // namespace adaptive

#include "selectImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_SELECT_HPP
