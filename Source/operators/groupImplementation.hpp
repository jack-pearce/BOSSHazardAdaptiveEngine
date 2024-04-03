#ifndef BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "hash_map/robin_map.h"
#include "tsl/robin_map.h"

namespace adaptive {

template <typename T> using Aggregator = std::function<T(const T, const T, bool)>;

/****************************** FOUNDATIONAL ALGORITHMS ********************************/

template <typename K1, typename... As>
inline void groupByHashAux(HA_tsl::robin_map<K1, std::tuple<As...>>& map, int& index, int n,
                           bool secondKey, const K1* keys1, const K1* keys2,
                           const As*... aggregates, Aggregator<As>... aggregator) {
  typename HA_tsl::robin_map<K1, std::tuple<As...>>::iterator it;

  // TODO compiler optimise out lambda for !secondKey?, if not then can have two loops
  auto getKey = [secondKey, keys1, keys2](int index) {
    if(!secondKey) {
      return keys1[index];
    }
    return (keys2[index] << 8) | keys1[index];
  };

  int startingIndex = index;
  for(; index < startingIndex + n; ++index) {
    auto key = getKey(index);
    it = map.find(key);
    if(it != map.end()) {
      it.value() = std::apply(
          [&](auto&&... args) {
            return std::make_tuple(aggregator(args, aggregates[index], false)...);
          },
          std::move(it->second));
    } else {
      map.insert({key, std::make_tuple(aggregator(0, aggregates[index], true)...)});
    }
  }
}

template <typename K1, typename... As>
std::vector<ExpressionSpanArguments>
groupByHash(int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
            ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
            Aggregator<As>... aggregators) {
  static_assert(std::is_integral<std::remove_cv_t<K1>>::value,
                "Key column must be an integer type");
  int initialSize = std::max(static_cast<int>(2.5 * cardinality), 400000);
  int index;

  HA_tsl::robin_map<std::remove_cv_t<K1>, std::tuple<std::remove_cv_t<As>...>> map(initialSize);

  for(size_t i = 0; i < keySpans1.size(); ++i) {
    auto& keySpan1 = std::get<Span<K1>>(keySpans1.at(i));
    auto& keySpan2 = secondKey ? std::get<Span<K1>>(keySpans2.at(i)) : keySpan1;
    index = 0;
    groupByHashAux<std::remove_cv_t<K1>, std::remove_cv_t<As>...>(
        map, index, keySpan1.size(), secondKey, &(keySpan1[0]), &(keySpan2[0]),
        &(typedAggCols[i][0])..., aggregators...);
  }

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);
  std::vector<std::remove_cv_t<K1>> keys;
  keys.reserve(map.size());
  std::optional<std::vector<std::remove_cv_t<K1>>> keys2;
  if(secondKey) {
    keys2 = std::make_optional<std::vector<std::remove_cv_t<K1>>>();
    keys2->reserve(map.size());
  }
  std::tuple<std::vector<std::remove_cv_t<As>>...> valueVectors;
  std::apply([&](auto&&... vec) { (vec.reserve(map.size()), ...); }, valueVectors);

  auto addKeys = [&keys, &keys2, secondKey](auto key) mutable {
    if(!secondKey) {
      keys.push_back(key);
    } else {
      keys.push_back(key & 0xFF);
      keys2->push_back(key >> 8);
    }
  };

  for(const auto& pair : map) {
    addKeys(pair.first);
    auto& tuple = pair.second;
    [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      ((std::get<Is>(valueVectors).push_back(std::get<Is>(tuple))), ...);
    }(std::make_index_sequence<sizeof...(As)>());
  }

  result.emplace_back(Span<std::remove_cv_t<K1>>(std::move(keys)));
  if(secondKey)
    result.emplace_back(Span<std::remove_cv_t<K1>>(std::move(*keys2)));
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(valueVectors)))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

/************************************ SINGLE-THREADED **********************************/

/************************** SINGLE-THREADED FOR MULTI-THREADED *************************/

/************************************ MULTI-THREADED ***********************************/

/********************************** UTILITY FUNCTIONS **********************************/

template <typename... As>
std::vector<ExpressionSpanArguments> groupNoKeys(std::vector<Span<As>>&&... typedAggCols,
                                                 Aggregator<As>... aggregators) {

  std::tuple<As...> resultValues = std::make_tuple(aggregators(0, typedAggCols[0][0], true)...);

  std::vector<size_t> sizes;
  bool sizesUpdated = false;
  (
      [&] {
        if(!sizesUpdated) {
          sizes.reserve(typedAggCols.size());
          for(auto& typedAggSpan : typedAggCols) {
            sizes.push_back(typedAggSpan.size());
          }
          sizesUpdated = true;
        }
      }(),
      ...);

  for(size_t i = 1; i < sizes[0]; ++i) {
    resultValues = std::apply(
        [&](auto&&... args) {
          return std::make_tuple(aggregators(args, typedAggCols[0][i], false)...);
        },
        std::move(resultValues));
  }

  for(size_t i = 1; i < sizes.size(); ++i) {
    for(size_t j = 0; j < sizes[i]; ++j) {
      resultValues = std::apply(
          [&](auto&&... args) {
            return std::make_tuple(aggregators(args, typedAggCols[i][j], false)...);
          },
          std::move(resultValues));
    }
  }

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As));
  std::tuple<std::vector<std::remove_cv_t<As>>...> valueVectors;

  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    ((std::get<Is>(valueVectors).push_back(std::get<Is>(resultValues))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(valueVectors)))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

/*********************************** ENTRY FUNCTION ************************************/

template <typename K1, typename... As>
std::vector<ExpressionSpanArguments>
group(Group implementation, int cardinality, int numKeys, ExpressionSpanArguments&& keySpans1,
      ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
      Aggregator<As>... aggregators) {
  assert(numKeys >= 0 && numKeys <= 2);
  if(numKeys == 0) {
    return groupNoKeys<As...>(std::move(typedAggCols)..., aggregators...);
  }
  switch(implementation) {
  case Group::Hash:
    return groupByHash<K1, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                  std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
  case Group::Sort:
    throw std::runtime_error("Not yet implemented");
  case Group::GroupAdaptive:
    throw std::runtime_error("Not yet implemented");
  case Group::GroupAdaptiveParallel:
    throw std::runtime_error("Not yet implemented");
  default:
    throw std::runtime_error("Invalid selection of 'Group' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP