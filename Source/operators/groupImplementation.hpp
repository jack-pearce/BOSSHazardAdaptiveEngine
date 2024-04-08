#ifndef BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <stdexcept>

#include "hash_map/robin_map.h"
#include "tsl/robin_map.h"

#define DEBUG

namespace adaptive {

constexpr int BITS_PER_GROUPBY_RADIX_PASS = 8;
constexpr int DEFAULT_GROUPBY_RESULT_CARDINALITY = 100;

template <typename T> using Aggregator = std::function<T(const T, const T, bool)>;

/********************************** UTILITY FUNCTIONS **********************************/

inline int getGroupResultCardinality() {
  int cardinality = DEFAULT_GROUPBY_RESULT_CARDINALITY;
  char* cardinalityStr = std::getenv("GROUP_RESULT_CARDINALITY");
  if(cardinalityStr != nullptr) {
    cardinality = std::atoi(cardinalityStr);
#ifdef DEBUG
    std::cout << "Read 'GROUP' result cardinality environment variable value of: " << cardinality
              << std::endl;
#endif
    return cardinality;
  }
#ifdef DEBUG
  std::cout << "'GROUP' result cardinality environment variable not set, using default value: "
            << cardinality << std::endl;
#endif
  return cardinality;
}

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

/****************************** FOUNDATIONAL ALGORITHMS ********************************/

template <typename K, typename... As>
inline void groupByHashAux(HA_tsl::robin_map<K, std::tuple<As...>>& map, int& index, int n,
                           bool secondKey, const K* keys1, const K* keys2, const As*... aggregates,
                           Aggregator<As>... aggregator) {
  typename HA_tsl::robin_map<K, std::tuple<As...>>::iterator it;

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

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupByHash(int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
            ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
            Aggregator<As>... aggregators) {
  int initialSize = std::max(static_cast<int>(2.5 * cardinality), 400000);
  int index;

  HA_tsl::robin_map<std::remove_cv_t<K>, std::tuple<std::remove_cv_t<As>...>> map(initialSize);

  for(size_t i = 0; i < keySpans1.size(); ++i) {
    auto& keySpan1 = std::get<Span<K>>(keySpans1.at(i));
    auto& keySpan2 = secondKey ? std::get<Span<K>>(keySpans2.at(i)) : keySpan1;
    index = 0;
    groupByHashAux<std::remove_cv_t<K>, std::remove_cv_t<As>...>(
        map, index, keySpan1.size(), secondKey, &(keySpan1[0]), &(keySpan2[0]),
        &(typedAggCols[i][0])..., aggregators...);
  }

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);
  std::vector<std::remove_cv_t<K>> keys;
  keys.reserve(map.size());
  std::optional<std::vector<std::remove_cv_t<K>>> keys2;
  if(secondKey) {
    keys2 = std::make_optional<std::vector<std::remove_cv_t<K>>>();
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

  result.emplace_back(Span<std::remove_cv_t<K>>(std::move(keys)));
  if(secondKey)
    result.emplace_back(Span<std::remove_cv_t<K>>(std::move(*keys2)));
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(valueVectors)))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

template <typename K, typename... As>
void groupBySortAggPassOnly(size_t n, K* keys, std::tuple<As...>* payloads,
                            std::vector<std::remove_cv_t<K>>& resultKeys,
                            std::vector<std::remove_cv_t<K>>& resultKeys2,
                            std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors,
                            bool secondKey, Aggregator<As>... aggregators) {
  auto addKeys = [&resultKeys, &resultKeys2, secondKey](auto key) mutable {
    if(!secondKey) {
      resultKeys.push_back(key);
    } else {
      resultKeys.push_back(key & 0xFF);
      resultKeys2.push_back(key >> 8);
    }
  };

  size_t i = 0;
  while(i < n) {
    auto key = keys[i];
    auto tuple = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      return std::make_tuple(aggregators(0, std::get<Is>(payloads[i]), true)...);
    }(std::make_index_sequence<sizeof...(As)>());
    ++i;
    while(keys[i] == key) {
      tuple = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        return std::make_tuple(
            aggregators(std::get<Is>(tuple), std::get<Is>(payloads[i]), false)...);
      }(std::make_index_sequence<sizeof...(As)>());
      ++i;
    }
    addKeys(key);
    [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(tuple))), ...);
    }(std::make_index_sequence<sizeof...(As)>());
  }
}

template <typename K, typename... As>
void groupBySortFinalPassAndAgg(
    size_t n, K* keys, std::tuple<As...>* payloads, std::vector<std::remove_cv_t<K>>& resultKeys,
    std::vector<std::remove_cv_t<K>>& resultKeys2,
    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors, bool secondKey,
    int msbPosition, Aggregator<As>... aggregators) {

  static bool bucketEntryPresent[1 << BITS_PER_GROUPBY_RADIX_PASS];
  static std::tuple<As...> payloadAggs[1 << BITS_PER_GROUPBY_RADIX_PASS];
  std::fill(std::begin(bucketEntryPresent), std::end(bucketEntryPresent), false);

  auto addKeys = [&resultKeys, &resultKeys2, secondKey](auto key) mutable {
    if(!secondKey) {
      resultKeys.push_back(key);
    } else {
      resultKeys.push_back(key & 0xFF);
      resultKeys2.push_back(key >> 8);
    }
  };

  size_t i;
  int radixBits = msbPosition;
  size_t numBuckets = 1 << radixBits;
  int mask = static_cast<int>(numBuckets) - 1;

  for(i = 0; i < n; i++) {
    auto keyLowerBits = keys[i] & mask;
    payloadAggs[keyLowerBits] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      return std::make_tuple(aggregators(std::get<Is>(payloadAggs[keyLowerBits]),
                                         std::get<Is>(payloads[i]),
                                         !bucketEntryPresent[keyLowerBits])...);
    }(std::make_index_sequence<sizeof...(As)>());
    bucketEntryPresent[keyLowerBits] = true;
  }

  int valuePrefix = keys[0] & ~mask;

  for(i = 0; i < numBuckets; i++) {
    if(bucketEntryPresent[i]) {
      addKeys(valuePrefix | i);
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloadAggs[i]))), ...);
      }(std::make_index_sequence<sizeof...(As)>());
    }
  }
}

template <typename K, typename... As>
void groupBySortAux(size_t n, std::vector<int>& buckets, K* keys, std::tuple<As...>* payloads,
                    K* keysBuffer, std::tuple<As...>* payloadsBuffer,
                    std::vector<std::remove_cv_t<K>>& resultKeys,
                    std::vector<std::remove_cv_t<K>>& resultKeys2,
                    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors,
                    bool secondKey, int msbPosition, Aggregator<As>... aggregators) {
  size_t i;
  int radixBits = BITS_PER_GROUPBY_RADIX_PASS;
  size_t numBuckets = 1 << radixBits;
  int mask = static_cast<int>(numBuckets) - 1;
  int shifts = msbPosition - radixBits;

  for(i = 0; i < n; i++) {
    buckets[1 + ((keys[i] >> shifts) & mask)]++;
  }

  for(i = 2; i <= numBuckets; i++) {
    buckets[i] += buckets[i - 1];
  }

  std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);

  for(i = 0; i < n; i++) {
    K key = keys[i];
    auto index = buckets[(key >> shifts) & mask]++;
    keysBuffer[index] = key;
    payloadsBuffer[index] = payloads[i];
  }

  std::fill(buckets.begin(), buckets.end(), 0);
  msbPosition -= radixBits;

  if(msbPosition <= BITS_PER_GROUPBY_RADIX_PASS) {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortFinalPassAndAgg<K, As...>(
            partitions[i] - prevPartitionEnd, prevPartitionEnd + keysBuffer,
            prevPartitionEnd + payloadsBuffer, resultKeys, resultKeys2, resultValueVectors,
            secondKey, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  } else {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortAux<K, As...>(
            partitions[i] - prevPartitionEnd, buckets, prevPartitionEnd + keysBuffer,
            prevPartitionEnd + payloadsBuffer, prevPartitionEnd + keys, prevPartitionEnd + payloads,
            resultKeys, resultKeys2, resultValueVectors, secondKey, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  }
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupBySort(int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
            ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
            Aggregator<As>... aggregators) {

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);
  std::vector<std::remove_cv_t<K>> resultKeys;
  resultKeys.reserve(cardinality);
  std::vector<std::remove_cv_t<K>> resultKeys2;
  if(secondKey) {
    resultKeys2.reserve(cardinality);
  }
  std::tuple<std::vector<std::remove_cv_t<As>>...> resultValueVectors;
  std::apply([&](auto&&... vec) { (vec.reserve(cardinality), ...); }, resultValueVectors);

  size_t n = 0;
  int msbPosition = 0;

  if(secondKey) {
    msbPosition = 16;
    for(const auto& untypedSpan : keySpans1) {
      auto& keySpan1 = std::get<Span<K>>(untypedSpan);
      n += keySpan1.size();
    }
  } else {
    K largest = 0;
    for(auto& untypedSpan : keySpans1) {
      auto& keySpan1 = std::get<Span<K>>(untypedSpan);
      n += keySpan1.size();
      for(const auto& value : keySpan1) {
        if(value > largest) {
          largest = value;
        }
      }
    }
    while(largest != 0) {
      largest >>= 1;
      msbPosition++;
    }
  }

  std::vector<int> buckets(1 + (1 << BITS_PER_GROUPBY_RADIX_PASS), 0);
  auto keysPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keys = keysPtr.get();
  auto payloadsPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloads = payloadsPtr.get();
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keysBuffer = keysBufferPtr.get();
  auto payloadsBufferPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloadsBuffer = payloadsBufferPtr.get();

  size_t i;
  int radixBits = std::min(msbPosition, BITS_PER_GROUPBY_RADIX_PASS);
  size_t numBuckets = 1 << radixBits;
  int mask = static_cast<int>(numBuckets) - 1;
  int shifts = msbPosition - radixBits;

  if(!secondKey) {
    for(auto& untypedSpan : keySpans1) {
      for(auto& key : std::get<Span<K>>(untypedSpan)) {
        buckets[1 + ((key >> shifts) & mask)]++;
      }
    }
  } else {
    for(size_t spanNum = 0; spanNum < keySpans1.size(); ++spanNum) {
      auto& keySpan1 = std::get<Span<K>>(keySpans1.at(spanNum));
      auto& keySpan2 = std::get<Span<K>>(keySpans2.at(spanNum));
      for(size_t index = 0; index < keySpan1.size(); ++index) {
        buckets[1 + ((((keySpan2[index] << 8) | keySpan1[index]) >> shifts) & mask)]++;
      }
    }
  }

  for(i = 2; i <= numBuckets; i++) {
    buckets[i] += buckets[i - 1];
  }
  std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);

  if(!secondKey) {
    for(size_t spanNum = 0; spanNum < keySpans1.size(); ++spanNum) {
      auto& keySpan1 = std::get<Span<K>>(keySpans1.at(spanNum));
      for(size_t spanIndex = 0; spanIndex < keySpan1.size(); ++spanIndex) {
        K key = keySpan1[spanIndex];
        auto index = buckets[(key >> shifts) & mask]++;
        keys[index] = key;
        payloads[index] = std::make_tuple(typedAggCols[spanNum][spanIndex]...);
      }
    }
  } else {
    for(size_t spanNum = 0; spanNum < keySpans1.size(); ++spanNum) {
      auto& keySpan1 = std::get<Span<K>>(keySpans1.at(spanNum));
      auto& keySpan2 = std::get<Span<K>>(keySpans2.at(spanNum));
      for(size_t spanIndex = 0; spanIndex < keySpan1.size(); ++spanIndex) {
        K key = (keySpan2[spanIndex] << 8) | keySpan1[spanIndex];
        auto index = buckets[(key >> shifts) & mask]++;
        keys[index] = key;
        payloads[index] = std::make_tuple(typedAggCols[spanNum][spanIndex]...);
      }
    }
  }

  msbPosition -= radixBits;
  if(msbPosition == 0) {
    groupBySortAggPassOnly<K, As...>(n, keys, payloads, resultKeys, resultKeys2, resultValueVectors,
                                     secondKey, aggregators...);
  } else if(msbPosition <= BITS_PER_GROUPBY_RADIX_PASS) {
    std::fill(buckets.begin(), buckets.end(), 0);
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortFinalPassAndAgg<K, As...>(
            partitions[i] - prevPartitionEnd, prevPartitionEnd + keys, prevPartitionEnd + payloads,
            resultKeys, resultKeys2, resultValueVectors, secondKey, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  } else {
    std::fill(buckets.begin(), buckets.end(), 0);
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortAux<K, As...>(partitions[i] - prevPartitionEnd, buckets, prevPartitionEnd + keys,
                                 prevPartitionEnd + payloads, prevPartitionEnd + keysBuffer,
                                 prevPartitionEnd + payloadsBuffer, resultKeys, resultKeys2,
                                 resultValueVectors, secondKey, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  }

  result.emplace_back(Span<std::remove_cv_t<K>>(std::move(resultKeys)));
  if(secondKey)
    result.emplace_back(Span<std::remove_cv_t<K>>(std::move(resultKeys2)));
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(resultValueVectors)))),
     ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

/************************************ SINGLE-THREADED **********************************/

/************************** SINGLE-THREADED FOR MULTI-THREADED *************************/

/************************************ MULTI-THREADED ***********************************/

/*********************************** ENTRY FUNCTION ************************************/

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
group(Group implementation, int numKeys, ExpressionSpanArguments&& keySpans1,
      ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
      Aggregator<As>... aggregators) {
  assert(numKeys >= 0 && numKeys <= 2);
  int cardinality = getGroupResultCardinality();
  if(numKeys == 0) {
    return groupNoKeys<As...>(std::move(typedAggCols)..., aggregators...);
  }
  switch(implementation) {
  case Group::Hash:
    return groupByHash<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                 std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
  case Group::Sort:
    return groupBySort<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                 std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
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