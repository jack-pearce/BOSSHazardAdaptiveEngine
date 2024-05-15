#ifndef BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <queue>
#include <stdexcept>

#include "constants/machineConstants.hpp"
#include "hash_map/robin_map.h"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/utilities.hpp"

#define ADAPTIVITY_OUTPUT
#define DEBUG
#define USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1

namespace adaptive {

constexpr float PERCENT_INPUT_TO_TRACK = 0.001;             // 0.1%
constexpr float PERCENT_INPUT_IN_TRANSIENT_CHECK = 0.00005; // 0.005%
constexpr int TUPLES_IN_CACHE_MISS_CHECK = 75 * 1000;       // 75,000
constexpr float PERCENT_INPUT_BETWEEN_HASHING = 0.25;       // 25%

constexpr int FIRST_KEY_BITS = 8; // Bits for first key when there are two keys
constexpr int FIRST_KEY_MASK = static_cast<int>(1 << FIRST_KEY_BITS) - 1;
constexpr int BITS_PER_GROUP_RADIX_PASS = 8;
constexpr int DEFAULT_GROUP_RESULT_CARDINALITY = 100;
constexpr float HASHMAP_OVERALLOCATION_FACTOR = 2.5;

template <typename T> using Aggregator = std::function<T(const T, const T)>;

/********************************** UTILITY FUNCTIONS **********************************/

template <typename K, typename... As>
using AggregatedKeysAndPayload =
    std::tuple<std::vector<K>, std::vector<K>, std::tuple<std::vector<As>...>>;

template <typename K, typename... As> struct Section {
  size_t n;
  K* key1;
  K* key2;
  std::tuple<As*...> aggs;

  Section(size_t n_, K* k1, K* k2, std::tuple<As*...> aggs_)
      : n(n_), key1(k1), key2(k2), aggs(aggs_) {}
};

inline int getGroupResultCardinality() {
  int cardinality = DEFAULT_GROUP_RESULT_CARDINALITY;
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

  std::tuple<As...> resultValues = std::make_tuple(typedAggCols[0][0]...);

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
        [&](auto&&... args) { return std::make_tuple(aggregators(args, typedAggCols[0][i])...); },
        std::move(resultValues));
  }

  for(size_t i = 1; i < sizes.size(); ++i) {
    for(size_t j = 0; j < sizes[i]; ++j) {
      resultValues = std::apply(
          [&](auto&&... args) { return std::make_tuple(aggregators(args, typedAggCols[i][j])...); },
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

template <typename K, typename... As>
void sortByKeyAux(size_t start, size_t end, K* keys, std::tuple<As*...> payloads, K* keysBuffer,
                  std::tuple<As*...> payloadBuffers, std::vector<int>& buckets, int msbPosition,
                  bool copyRequired) {
  size_t i;
  int radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
  int shifts = msbPosition - radixBits;
  size_t numBuckets = 1 << radixBits;
  unsigned int mask = static_cast<int>(numBuckets) - 1;

  for(i = start; i < end; i++) {
    buckets[1 + ((keys[i] >> shifts) & mask)]++;
  }

  for(i = 2; i <= numBuckets; i++) {
    buckets[i] += buckets[i - 1];
  }

  std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);

  for(i = start; i < end; i++) {
    K key = keys[i];
    auto index = start + buckets[(key >> shifts) & mask]++;
    keysBuffer[index] = key;
    [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      ((std::get<Is>(payloadBuffers)[index] = std::get<Is>(payloads)[i]), ...);
    }(std::make_index_sequence<sizeof...(As)>());
  }

  std::fill(buckets.begin(), buckets.end(), 0);
  msbPosition -= radixBits;

  if(msbPosition == 0) {
    if(copyRequired) {
      std::memcpy(start + keys, start + keysBuffer, (end - start) * sizeof(K));
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::memcpy(start + std::get<Is>(payloads), start + std::get<Is>(payloadBuffers),
                      (end - start) * sizeof(std::tuple_element_t<Is, std::tuple<As...>>))),
         ...);
      }(std::make_index_sequence<sizeof...(As)>());
    }
  } else {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        sortByKeyAux<K, As...>(start + prevPartitionEnd, start + partitions[i], keysBuffer,
                               payloadBuffers, keys, payloads, buckets, msbPosition, !copyRequired);
      }
      prevPartitionEnd = partitions[i];
    }
  }
}

template <typename K, typename... As>
void sortByKey(int n, K largestKey, K* keys, std::tuple<As*...> payloads) {
  int msbPosition = 0;
  while(largestKey != 0) {
    largestKey >>= 1;
    msbPosition++;
  }

  std::vector<int> buckets(1 + (1 << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keysBuffer = keysBufferPtr.get();
  std::tuple<std::unique_ptr<As[]>...> payloadBufferPtrs =
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        return std::make_tuple(
            std::make_unique_for_overwrite<std::tuple_element_t<Is, std::tuple<As...>>[]>(n)...);
      }(std::make_index_sequence<sizeof...(As)>());
  std::tuple<As*...> payloadBuffers = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    return std::make_tuple(std::get<Is>(payloadBufferPtrs).get()...);
  }(std::make_index_sequence<sizeof...(As)>());

  sortByKeyAux<K, As...>(0, n, keys, payloads, keysBuffer, payloadBuffers, buckets, msbPosition,
                         true);
}

/****************************** FOUNDATIONAL ALGORITHMS ********************************/

template <typename K, typename... As>
inline void groupByHashAux(HA_tsl::robin_map<K, std::tuple<As...>>& map, int& index, int n,
                           bool secondKey, const K* keys1, const K* keys2, const As*... aggregates,
                           Aggregator<As>... aggregators) {
  typename HA_tsl::robin_map<K, std::tuple<As...>>::iterator it;

  auto getKey = [secondKey, keys1, keys2](int index) {
    if(!secondKey) {
      return keys1[index];
    }
    return (keys2[index] << FIRST_KEY_BITS) | keys1[index];
  };

  int startingIndex = index;
  for(; index < startingIndex + n; ++index) {
    auto key = getKey(index);
    it = map.find(key);
    if(it != map.end()) {
      it.value() = std::apply(
          [&](auto&&... args) { return std::make_tuple(aggregators(args, aggregates[index])...); },
          std::move(it->second));
    } else {
      map.insert({key, std::make_tuple(aggregates[index]...)});
    }
  }
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupByHash(int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
            ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
            Aggregator<As>... aggregators) {
  int initialSize = std::max(
      static_cast<int>(HASHMAP_OVERALLOCATION_FACTOR * static_cast<float>(cardinality)), 400000);
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
      keys.push_back(key & FIRST_KEY_MASK);
      keys2->push_back(key >> FIRST_KEY_BITS);
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
                            bool secondKey, bool splitKeysInResult, Aggregator<As>... aggregators) {
  auto addKeys = [&resultKeys, &resultKeys2, secondKey, splitKeysInResult](auto key) mutable {
    if(secondKey && splitKeysInResult) {
      resultKeys.push_back(key & FIRST_KEY_MASK);
      resultKeys2.push_back(key >> FIRST_KEY_BITS);
    } else {
      resultKeys.push_back(key);
    }
  };

  size_t i = 0;
  while(i < n) {
    auto key = keys[i];
    auto tuple = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      return std::make_tuple(std::get<Is>(payloads[i])...);
    }(std::make_index_sequence<sizeof...(As)>());
    ++i;
    while(keys[i] == key) {
      tuple = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        return std::make_tuple(aggregators(std::get<Is>(tuple), std::get<Is>(payloads[i]))...);
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
    bool splitKeysInResult, int msbPosition, Aggregator<As>... aggregators) {

  thread_local static bool bucketEntryPresent[1 << BITS_PER_GROUP_RADIX_PASS];
  thread_local static std::tuple<As...> payloadAggs[1 << BITS_PER_GROUP_RADIX_PASS];
  std::fill(std::begin(bucketEntryPresent), std::end(bucketEntryPresent), false);

  auto addKeys = [&resultKeys, &resultKeys2, secondKey, splitKeysInResult](auto key) mutable {
    if(secondKey && splitKeysInResult) {
      resultKeys.push_back(key & FIRST_KEY_MASK);
      resultKeys2.push_back(key >> FIRST_KEY_BITS);
    } else {
      resultKeys.push_back(key);
    }
  };

  size_t i;
  int radixBits = msbPosition;
  size_t numBuckets = 1 << radixBits;
  int mask = static_cast<int>(numBuckets) - 1;

  for(i = 0; i < n; i++) {
    auto keyLowerBits = keys[i] & mask;
    payloadAggs[keyLowerBits] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      if(bucketEntryPresent[keyLowerBits]) {
        return std::make_tuple(
            aggregators(std::get<Is>(payloadAggs[keyLowerBits]), std::get<Is>(payloads[i]))...);
      } else {
        return std::make_tuple(std::get<Is>(payloads[i])...);
      }
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
                    bool secondKey, bool splitKeysInResult, int msbPosition,
                    Aggregator<As>... aggregators) {
  size_t i;
  int radixBits = BITS_PER_GROUP_RADIX_PASS;
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

  if(msbPosition <= BITS_PER_GROUP_RADIX_PASS) {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortFinalPassAndAgg<K, As...>(
            partitions[i] - prevPartitionEnd, prevPartitionEnd + keysBuffer,
            prevPartitionEnd + payloadsBuffer, resultKeys, resultKeys2, resultValueVectors,
            secondKey, splitKeysInResult, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  } else {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortAux<K, As...>(partitions[i] - prevPartitionEnd, buckets,
                                 prevPartitionEnd + keysBuffer, prevPartitionEnd + payloadsBuffer,
                                 prevPartitionEnd + keys, prevPartitionEnd + payloads, resultKeys,
                                 resultKeys2, resultValueVectors, secondKey, splitKeysInResult,
                                 msbPosition, aggregators...);
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
  K largest = std::numeric_limits<K>::lowest();

  if(secondKey) {
    for(auto& untypedSpan : keySpans2) {
      auto& keySpan2 = std::get<Span<K>>(untypedSpan);
      n += keySpan2.size();
      for(const auto& value : keySpan2) {
        largest = std::max(largest, value); // Bits in second key
      }
    }
  } else {
    for(auto& untypedSpan : keySpans1) {
      auto& keySpan1 = std::get<Span<K>>(untypedSpan);
      n += keySpan1.size();
      for(const auto& value : keySpan1) {
        largest = std::max(largest, value); // Bits in first key
      }
    }
  }
  while(largest != 0) {
    largest >>= 1;
    msbPosition++;
  }
  if(secondKey) {
    msbPosition += FIRST_KEY_BITS; // Add bits in first key for combined key
  }

  std::vector<int> buckets(1 + (1 << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keys = keysPtr.get();
  auto payloadsPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloads = payloadsPtr.get();
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keysBuffer = keysBufferPtr.get();
  auto payloadsBufferPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloadsBuffer = payloadsBufferPtr.get();

  size_t i;
  int radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
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
        buckets[1 + ((((keySpan2[index] << FIRST_KEY_BITS) | keySpan1[index]) >> shifts) & mask)]++;
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
        K key = (keySpan2[spanIndex] << FIRST_KEY_BITS) | keySpan1[spanIndex];
        auto index = buckets[(key >> shifts) & mask]++;
        keys[index] = key;
        payloads[index] = std::make_tuple(typedAggCols[spanNum][spanIndex]...);
      }
    }
  }

  msbPosition -= radixBits;
  if(msbPosition == 0) {
    groupBySortAggPassOnly<K, As...>(n, keys, payloads, resultKeys, resultKeys2, resultValueVectors,
                                     secondKey, true, aggregators...);
  } else if(msbPosition <= BITS_PER_GROUP_RADIX_PASS) {
    std::fill(buckets.begin(), buckets.end(), 0);
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortFinalPassAndAgg<K, As...>(partitions[i] - prevPartitionEnd,
                                             prevPartitionEnd + keys, prevPartitionEnd + payloads,
                                             resultKeys, resultKeys2, resultValueVectors, secondKey,
                                             true, msbPosition, aggregators...);
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
                                 resultValueVectors, secondKey, true, msbPosition, aggregators...);
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

class MonitorGroup {
public:
  explicit MonitorGroup(const long_long* lastLevelCacheMisses_,
                        float pageFaultDecreaseRatePerTuple_, float tuplesPerLastLevelCacheMiss_)
      : lastLevelCacheMisses(lastLevelCacheMisses_),
        pageFaultDecreaseRatePerTuple(pageFaultDecreaseRatePerTuple_),
        tuplesPerLastLevelCacheMiss(tuplesPerLastLevelCacheMiss_) {}

  [[nodiscard]] inline bool
  robustnessIncreaseRequiredBasedOnPageFaults(double pageFaultsPerTuple,
                                              double measuredPageFaultDecreaseRatePerTuple) const {
    // High rate of page faults which is not decreasing, indicating a large working set size
    return pageFaultsPerTuple > 1 &&
           measuredPageFaultDecreaseRatePerTuple < pageFaultDecreaseRatePerTuple;
  }

  inline bool robustnessIncreaseRequiredBasedOnCacheMisses(int tuplesProcessed) {
#ifdef DEBUG
    bool result = (static_cast<float>(tuplesProcessed) /
                   static_cast<float>(*lastLevelCacheMisses)) < tuplesPerLastLevelCacheMiss;
    if(result) {
      std::cout << "Increasing robustness (" << tuplesProcessed << " tuples / "
                << *lastLevelCacheMisses << " cache misses): "
                << (static_cast<float>(tuplesProcessed) / static_cast<float>(*lastLevelCacheMisses))
                << std::endl;
    }
    return result;
#else
    return (static_cast<float>(tuplesProcessed) / static_cast<float>(*lastLevelCacheMisses)) <
           tuplesPerLastLevelCacheMiss;
#endif
  }

private:
  const long_long* lastLevelCacheMisses;
  float pageFaultDecreaseRatePerTuple;
  float tuplesPerLastLevelCacheMiss;
};

template <typename K, typename... As>
inline void groupByAdaptiveAuxHash(HA_tsl::robin_map<K, std::tuple<As...>>& map, K& largestKey,
                                   std::pair<K, std::tuple<As...>>** mapPtrs, int& mapPtrsSize,
                                   const int entriesInMapToTrack, int n, bool secondKey,
                                   const K* keys1, const K* keys2, const As*... aggregates,
                                   Aggregator<As>... aggregators) {
  typename HA_tsl::robin_map<K, std::tuple<As...>>::iterator it;

  auto getKey = [secondKey, keys1, keys2](int index) {
    if(!secondKey) {
      return keys1[index];
    }
    return (keys2[index] << FIRST_KEY_BITS) | keys1[index];
  };

  int index = 0;

  while(mapPtrsSize < entriesInMapToTrack && index < n) {
    auto key = getKey(index);
    it = map.find(key);
    if(it != map.end()) {
      it.value() = std::apply(
          [&](auto&&... args) { return std::make_tuple(aggregators(args, aggregates[index])...); },
          std::move(it->second));
    } else {
      auto insertionResult = map.insert({key, std::make_tuple(aggregates[index]...)});
      mapPtrs[mapPtrsSize++] =
          const_cast<std::pair<K, std::tuple<As...>>*>(&(*insertionResult.first));
      largestKey = std::max(largestKey, key);
    }
    ++index;
  }

  for(; index < n; ++index) {
    auto key = getKey(index);
    it = map.find(key);
    if(it != map.end()) {
      it.value() = std::apply(
          [&](auto&&... args) { return std::make_tuple(aggregators(args, aggregates[index])...); },
          std::move(it->second));
    } else {
      map.insert({key, std::make_tuple(aggregates[index]...)});
      largestKey = std::max(largestKey, key);
    }
  }
}

template <typename K, typename... As>
AggregatedKeysAndPayload<K, As...>
groupByAdaptiveAuxSort(HA_tsl::robin_map<K, std::tuple<As...>>& map, K& largestKey,
                       std::pair<K, std::tuple<As...>>** mapPtrs, const int entriesInMapToTrack,
                       size_t n, int cardinality, std::vector<Section<K, As...>>& sections,
                       bool secondKey, bool splitKeysInResult, Aggregator<As>... aggregators) {

  std::vector<std::remove_cv_t<K>> resultKeys;
  resultKeys.reserve(cardinality);
  std::vector<std::remove_cv_t<K>> resultKeys2;
  if(secondKey) {
    resultKeys2.reserve(cardinality);
  }
  std::tuple<std::vector<std::remove_cv_t<As>>...> resultValueVectors;
  std::apply([&](auto&&... vec) { (vec.reserve(cardinality), ...); }, resultValueVectors);

  size_t i;
  int msbPosition = 0;

  if(secondKey) {
    K largestSecondKey = std::numeric_limits<K>::lowest();
    for(const auto& section : sections) {
      K* keyPtr = section.key2;
      for(i = 0; i < section.n; ++i) {
        largestSecondKey = std::max(largestSecondKey, keyPtr[i]); // Bits in second key
      }
    } // Bits in combined key
    largestKey = std::max(largestKey, largestSecondKey << FIRST_KEY_BITS);
  } else {
    for(const auto& section : sections) {
      K* keyPtr = section.key1;
      for(i = 0; i < section.n; ++i) {
        largestKey = std::max(largestKey, keyPtr[i]); // Bits in first key
      }
    }
  }
  while(largestKey != 0) {
    largestKey >>= 1;
    msbPosition++;
  }

  std::vector<int> buckets(1 + (1 << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keys = keysPtr.get();
  auto payloadsPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloads = payloadsPtr.get();
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n);
  auto* keysBuffer = keysBufferPtr.get();
  auto payloadsBufferPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n);
  auto* payloadsBuffer = payloadsBufferPtr.get();

  int radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
  size_t numBuckets = 1 << radixBits;
  auto mask = static_cast<int>(numBuckets) - 1;
  int shifts = msbPosition - radixBits;

  if(!secondKey) {
    for(const auto& section : sections) {
      K* keyPtr = section.key1;
      for(i = 0; i < section.n; ++i) {
        buckets[1 + ((keyPtr[i] >> shifts) & mask)]++;
      }
    }
  } else {
    for(const auto& section : sections) {
      K* keyPtr1 = section.key1;
      K* keyPtr2 = section.key2;
      for(i = 0; i < section.n; ++i) {
        buckets[1 + ((((keyPtr2[i] << FIRST_KEY_BITS) | keyPtr1[i]) >> shifts) & mask)]++;
      }
    }
  }

  std::vector<std::pair<K, std::tuple<As...>>> mapEntries;
  if(map.size() <= static_cast<size_t>(entriesInMapToTrack)) {
    mapEntries.reserve(map.size());
    for(i = 0; i < map.size(); ++i) {
      mapEntries.push_back(*(mapPtrs[i]));
    }
    for(auto& mapEntry : mapEntries) {
      buckets[1 + ((mapEntry.first >> shifts) & mask)]++;
    }
  } else {
    for(auto it = map.begin(); it != map.end(); ++it) {
      buckets[1 + ((it->first >> shifts) & mask)]++;
    }
  }

  for(i = 2; i <= numBuckets; i++) {
    buckets[i] += buckets[i - 1];
  }
  std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);
  if(map.size() <= static_cast<size_t>(entriesInMapToTrack)) {
    for(auto& mapEntry : mapEntries) {
      K key = mapEntry.first;
      auto index = buckets[(key >> shifts) & mask]++;
      keys[index] = key;
      payloads[index] = std::move(mapEntry.second);
    }
  } else {
    for(auto it = map.begin(); it != map.end(); ++it) {
      K key = it->first;
      auto index = buckets[(key >> shifts) & mask]++;
      keys[index] = key;
      payloads[index] = std::move(it->second);
    }
  }
  if(!secondKey) {
    for(const auto& section : sections) {
      K* keyPtr1 = section.key1;
      for(i = 0; i < section.n; ++i) {
        K key = keyPtr1[i];
        auto index = buckets[(key >> shifts) & mask]++;
        keys[index] = key;
        payloads[index] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
          return std::make_tuple(std::get<Is>(section.aggs)[i]...);
        }(std::make_index_sequence<sizeof...(As)>());
      }
    }
  } else {
    for(const auto& section : sections) {
      K* keyPtr1 = section.key1;
      K* keyPtr2 = section.key2;
      for(i = 0; i < section.n; ++i) {
        K key = (keyPtr2[i] << FIRST_KEY_BITS) | keyPtr1[i];
        auto index = buckets[(key >> shifts) & mask]++;
        keys[index] = key;
        payloads[index] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
          return std::make_tuple(std::get<Is>(section.aggs)[i]...);
        }(std::make_index_sequence<sizeof...(As)>());
      }
    }
  }

  msbPosition -= radixBits;
  if(msbPosition == 0) {
    groupBySortAggPassOnly<K, As...>(n, keys, payloads, resultKeys, resultKeys2, resultValueVectors,
                                     secondKey, splitKeysInResult, aggregators...);
  } else if(msbPosition <= BITS_PER_GROUP_RADIX_PASS) {
    std::fill(buckets.begin(), buckets.end(), 0);
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortFinalPassAndAgg<K, As...>(partitions[i] - prevPartitionEnd,
                                             prevPartitionEnd + keys, prevPartitionEnd + payloads,
                                             resultKeys, resultKeys2, resultValueVectors, secondKey,
                                             splitKeysInResult, msbPosition, aggregators...);
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
                                 resultValueVectors, secondKey, splitKeysInResult, msbPosition,
                                 aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  }

  return std::make_tuple(resultKeys, resultKeys2, resultValueVectors);
}

template <typename K, typename... As>
AggregatedKeysAndPayload<K, As...>
groupByAdaptive(int dop, const std::vector<int>& spanSizes, int n, int outerIndex, int innerIndex,
                int cardinality, bool secondKey, bool splitKeysInResult, K& largestKey,
                const ExpressionSpanArguments& keySpans1, const ExpressionSpanArguments& keySpans2,
                const std::vector<Span<As>>&... typedAggCols, Aggregator<As>... aggregators) {

  int tuplesInTransientCheck =
      static_cast<int>(PERCENT_INPUT_IN_TRANSIENT_CHECK * static_cast<float>(n));
  int tuplesPerTransientCheckReading = static_cast<int>(std::ceil(tuplesInTransientCheck / 10.0));
  tuplesInTransientCheck = tuplesPerTransientCheckReading * 10;
  std::vector<int> tuplesPerReading;
  if(tuplesInTransientCheck < 1000 || tuplesInTransientCheck > n) {
    tuplesInTransientCheck = 0;
  } else {
    tuplesPerReading.reserve(10);
    for(int i = 1; i < 11; i++) {
      tuplesPerReading.push_back(tuplesPerTransientCheckReading * i);
    }
  }

  int tuplesBetweenHashing =
      std::max(1, static_cast<int>(PERCENT_INPUT_BETWEEN_HASHING * static_cast<float>(n)));

#ifdef DEBUG
  std::cout << "tuplesInTransientCheck: " << tuplesInTransientCheck << std::endl;
  std::cout << "tuplesInCacheMissCheck: " << TUPLES_IN_CACHE_MISS_CHECK << std::endl;
  std::cout << "tuplesBetweenHashing:   " << tuplesBetweenHashing << std::endl;
#endif

  int initialSize = std::max(
      static_cast<int>(HASHMAP_OVERALLOCATION_FACTOR * static_cast<float>(cardinality)), 400000);
  HA_tsl::robin_map<K, std::tuple<As...>> map(initialSize);

  int entriesInMapToTrack = static_cast<int>(PERCENT_INPUT_TO_TRACK * static_cast<float>(n));
  int mapPtrsSize = 0;
  auto mapPtrsRaii =
      std::make_unique_for_overwrite<std::pair<K, std::tuple<As...>>*[]>(entriesInMapToTrack);
  auto* mapPtrs = mapPtrsRaii.get();

  auto& eventSet = getThreadEventSet();
  long_long* baseEventPtr = eventSet.getCounterDiffsPtr();

  auto& constants = MachineConstants::getInstance();
  constexpr int numBytes = (sizeof(K) + ... + sizeof(As));
  std::string name1 =
      "Group_" + std::to_string(numBytes) + "B_" + std::to_string(dop) + "_dop_PageFaults";
  std::string name2 = "Group_" + std::to_string(numBytes) + "B_" + std::to_string(dop) + "_dop_LLC";
  auto pageFaultDecreaseRatePerTupleThreshold =
      static_cast<float>(constants.getMachineConstant(name1));
  auto tuplesPerLastLevelCacheMiss = static_cast<float>(constants.getMachineConstant(name2));
  auto monitor = MonitorGroup(baseEventPtr + EVENT::LAST_LEVEL_CACHE_MISSES,
                              pageFaultDecreaseRatePerTupleThreshold, tuplesPerLastLevelCacheMiss);

  std::vector<Section<K, As...>> sectionsToBeSorted;
  int elements = 0;
  int tuplesToProcess, tuplesInCheck;
  double pageFaultDecreaseRatePerTuple, pageFaultsPerTuple;

  int tuplesProcessed = 0;

  auto runGroupByHashOnSection = [&](int tuplesToProcess_) mutable {
    if(!secondKey) {
      groupByAdaptiveAuxHash<K, As...>(map, largestKey, mapPtrs, mapPtrsSize, entriesInMapToTrack,
                                       tuplesToProcess_, secondKey,
                                       &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
                                       &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
                                       &(typedAggCols[outerIndex][innerIndex])..., aggregators...);
    } else {
      groupByAdaptiveAuxHash<K, As...>(map, largestKey, mapPtrs, mapPtrsSize, entriesInMapToTrack,
                                       tuplesToProcess_, secondKey,
                                       &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
                                       &(std::get<Span<K>>(keySpans2.at(outerIndex))[innerIndex]),
                                       &(typedAggCols[outerIndex][innerIndex])..., aggregators...);
    }
  };

  auto addSectionToSectionsToBeSorted = [&](int tuplesToProcess_) mutable {
    if(!secondKey) {
      sectionsToBeSorted.emplace_back(Section<K, As...>(
          tuplesToProcess_, &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
          &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
          std::make_tuple(&(typedAggCols[outerIndex][innerIndex])...)));
    } else {
      sectionsToBeSorted.emplace_back(Section<K, As...>(
          tuplesToProcess_, &(std::get<Span<K>>(keySpans1.at(outerIndex))[innerIndex]),
          &(std::get<Span<K>>(keySpans2.at(outerIndex))[innerIndex]),
          std::make_tuple(&(typedAggCols[outerIndex][innerIndex])...)));
    }
  };

  while(tuplesProcessed < n) {

    bool performTransientCheck =
        tuplesInTransientCheck > 0 && n - tuplesProcessed > tuplesInTransientCheck;
    if(performTransientCheck) {
      tuplesInCheck = tuplesInTransientCheck;
      tuplesProcessed += tuplesInTransientCheck;
      std::vector<long_long> pageFaults;
      pageFaults.reserve(10);
      for(int i = 0; i < 10; i++) {
        tuplesToProcess = tuplesInTransientCheck / 10;
        eventSet.readCounters();
        //////////// GROUP BY HASH ON tuplesToProcess elements ////////////
        while(tuplesToProcess && (spanSizes[outerIndex] - innerIndex) <= tuplesToProcess) {
          runGroupByHashOnSection(spanSizes[outerIndex] - innerIndex);
          tuplesToProcess -= (spanSizes[outerIndex] - innerIndex);
          ++outerIndex;
          innerIndex = 0;
        }
        if(tuplesToProcess > 0) {
          runGroupByHashOnSection(tuplesToProcess);
          innerIndex += tuplesToProcess;
        }
        ///////////////////////////////////////////////////////////////////
        eventSet.readCountersAndUpdateDiff();
        pageFaults.push_back(*(baseEventPtr + EVENT::PAGE_FAULTS));
#ifdef DEBUG
        std::cout << "pageFaults: " << pageFaults.back() << std::endl;
#endif
      }
      pageFaultsPerTuple =
          static_cast<double>(pageFaults.back()) / static_cast<double>(tuplesPerReading[0]);
      pageFaultDecreaseRatePerTuple = std::abs(linearRegressionSlope(tuplesPerReading, pageFaults));
#ifdef DEBUG
      std::cout << "pageFaultsPerTuple: " << pageFaultsPerTuple << std::endl;
      std::cout << "pageFaultDecreaseRatePerTuple: " << pageFaultDecreaseRatePerTuple << std::endl;
#endif
    }

    if(performTransientCheck && monitor.robustnessIncreaseRequiredBasedOnPageFaults(
                                    pageFaultsPerTuple, pageFaultDecreaseRatePerTuple)) {
#ifdef ADAPTIVITY_OUTPUT
      std::cout << "Switched to sort after processing " << tuplesProcessed << " tuples"
                << std::endl;
#endif
      tuplesToProcess = std::min(tuplesBetweenHashing, n - tuplesProcessed);
      tuplesProcessed += tuplesToProcess;
      elements += tuplesToProcess;
      //////////// GROUP BY SORT ON tuplesToProcess elements ////////////
      while(tuplesToProcess && (spanSizes[outerIndex] - innerIndex) <= tuplesToProcess) {
        addSectionToSectionsToBeSorted(spanSizes[outerIndex] - innerIndex);
        tuplesToProcess -= (spanSizes[outerIndex] - innerIndex);
        ++outerIndex;
        innerIndex = 0;
      }
      if(tuplesToProcess > 0) {
        addSectionToSectionsToBeSorted(tuplesToProcess);
        innerIndex += tuplesToProcess;
      }
      ///////////////////////////////////////////////////////////////////
      continue;
    }

    // Cache warmup
    tuplesToProcess = std::min(TUPLES_IN_CACHE_MISS_CHECK, n - tuplesProcessed);
    tuplesProcessed += tuplesToProcess;
    //////////// GROUP BY HASH ON tuplesToProcess elements ////////////
    while(tuplesToProcess && (spanSizes[outerIndex] - innerIndex) <= tuplesToProcess) {
      runGroupByHashOnSection(spanSizes[outerIndex] - innerIndex);
      tuplesToProcess -= (spanSizes[outerIndex] - innerIndex);
      ++outerIndex;
      innerIndex = 0;
    }
    if(tuplesToProcess > 0) {
      runGroupByHashOnSection(tuplesToProcess);
      innerIndex += tuplesToProcess;
    }
    ///////////////////////////////////////////////////////////////////

    while(tuplesProcessed < n) {

      tuplesToProcess = std::min(TUPLES_IN_CACHE_MISS_CHECK, n - tuplesProcessed);
      tuplesInCheck = tuplesToProcess;
      tuplesProcessed += tuplesToProcess;
      eventSet.readCounters();
      //////////// GROUP BY HASH ON tuplesToProcess elements ////////////
      while(tuplesToProcess && (spanSizes[outerIndex] - innerIndex) <= tuplesToProcess) {
        runGroupByHashOnSection(spanSizes[outerIndex] - innerIndex);
        tuplesToProcess -= (spanSizes[outerIndex] - innerIndex);
        ++outerIndex;
        innerIndex = 0;
      }
      if(tuplesToProcess > 0) {
        runGroupByHashOnSection(tuplesToProcess);
        innerIndex += tuplesToProcess;
      }
      ///////////////////////////////////////////////////////////////////
      eventSet.readCountersAndUpdateDiff();

      if(monitor.robustnessIncreaseRequiredBasedOnCacheMisses(tuplesInCheck)) {
#ifdef ADAPTIVITY_OUTPUT
        std::cout << "Switched to sort at index " << tuplesProcessed << std::endl;
#endif
        tuplesToProcess = std::min(tuplesBetweenHashing, n - tuplesProcessed);
        tuplesProcessed += tuplesToProcess;
        elements += tuplesToProcess;
        //////////// GROUP BY SORT ON tuplesToProcess elements ////////////
        while(tuplesToProcess && (spanSizes[outerIndex] - innerIndex) <= tuplesToProcess) {
          addSectionToSectionsToBeSorted(spanSizes[outerIndex] - innerIndex);
          tuplesToProcess -= (spanSizes[outerIndex] - innerIndex);
          ++outerIndex;
          innerIndex = 0;
        }
        if(tuplesToProcess > 0) {
          addSectionToSectionsToBeSorted(tuplesToProcess);
          innerIndex += tuplesToProcess;
        }
        ///////////////////////////////////////////////////////////////////
        break;
      }
    }
  }

  if(sectionsToBeSorted.empty()) {

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

    auto addKeys = [&resultKeys, &resultKeys2, secondKey, splitKeysInResult](auto key) mutable {
      if(secondKey && splitKeysInResult) {
        resultKeys.push_back(key & FIRST_KEY_MASK);
        resultKeys2.push_back(key >> FIRST_KEY_BITS);
      } else {
        resultKeys.push_back(key);
      }
    };

    for(const auto& pair : map) {
      addKeys(pair.first);
      auto& tuple = pair.second;
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(tuple))), ...);
      }(std::make_index_sequence<sizeof...(As)>());
    }

    return std::make_tuple(resultKeys, resultKeys2, resultValueVectors);
  }

  elements += map.size();
  return groupByAdaptiveAuxSort<K, As...>(map, largestKey, mapPtrs, entriesInMapToTrack, elements,
                                          cardinality, sectionsToBeSorted, secondKey,
                                          splitKeysInResult, aggregators...);
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupByAdaptive(int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
                ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
                Aggregator<As>... aggregators) {
  K largestKey = std::numeric_limits<K>::lowest();
  int n = 0;
  std::vector<int> spanSizes;
  spanSizes.reserve(keySpans1.size());
  for(const auto& untypedSpan : keySpans1) {
    auto& keySpan1 = std::get<Span<K>>(untypedSpan);
    n += keySpan1.size();
    spanSizes.push_back(keySpan1.size());
  }

  auto resultVectors =
      groupByAdaptive<K, As...>(1, spanSizes, n, 0, 0, cardinality, secondKey, true, largestKey,
                                keySpans1, keySpans2, typedAggCols..., aggregators...);

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);

  result.emplace_back(Span<K>(std::move(std::get<0>(resultVectors))));
  if(secondKey)
    result.emplace_back(Span<K>(std::move(std::get<1>(resultVectors))));
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<As>(std::move(std::get<Is>(std::get<2>(resultVectors))))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

/************************************ MULTI-THREADED ***********************************/

template <typename K, typename... As>
void groupByAdaptiveParallelPerformMerge(std::mutex& resultsMutex, std::condition_variable& cv,
                                         AggregatedKeysAndPayload<K, As...>&& mergeInput1,
                                         AggregatedKeysAndPayload<K, As...>&& mergeInput2,
                                         std::queue<AggregatedKeysAndPayload<K, As...>>& results,
                                         Aggregator<As>... aggregators) {
  std::vector<K> keys1 = std::move(std::get<0>(mergeInput1));
  std::vector<K> keys2 = std::move(std::get<0>(mergeInput2));
  std::tuple<std::vector<As>...> payloads1 = std::move(std::get<2>(mergeInput1));
  std::tuple<std::vector<As>...> payloads2 = std::move(std::get<2>(mergeInput2));

  size_t n1 = keys1.size();
  size_t n2 = keys2.size();
  size_t maxOutputSize = n1 + n2;

  std::vector<K> resultKeys;
  resultKeys.reserve(maxOutputSize);
  std::vector<K> resultKeys2;
  std::tuple<std::vector<As>...> resultValueVectors;
  std::apply([&](auto&&... vec) { (vec.reserve(maxOutputSize), ...); }, resultValueVectors);

  size_t l = 0;
  size_t r = 0;

  while(l < n1 && r < n2) {
    if(keys1[l] < keys2[r]) {
      resultKeys.push_back(keys1[l]);
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloads1)[l])), ...);
      }(std::make_index_sequence<sizeof...(As)>());
      l++;
    } else if(keys2[r] < keys1[l]) {
      resultKeys.push_back(keys2[r]);
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloads2)[r])), ...);
      }(std::make_index_sequence<sizeof...(As)>());
      r++;
    } else {
      resultKeys.push_back(keys1[l]);
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors)
              .push_back(aggregators(std::get<Is>(payloads1)[l], std::get<Is>(payloads2)[r]))),
         ...);
      }(std::make_index_sequence<sizeof...(As)>());
      l++;
      r++;
    }
  }

  if(l < n1) {
    if(!resultKeys.empty() && resultKeys.back() == keys1[l]) {
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).back() =
              aggregators(std::get<Is>(resultValueVectors).back(), std::get<Is>(payloads1)[l])),
         ...);
      }(std::make_index_sequence<sizeof...(As)>());
      l++;
    }
    resultKeys.insert(resultKeys.end(), std::make_move_iterator(keys1.begin() + l),
                      std::make_move_iterator(keys1.end()));
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((std::get<Is>(resultValueVectors)
            .insert(std::get<Is>(resultValueVectors).end(),
                    std::make_move_iterator(std::get<Is>(payloads1).begin() + l),
                    std::make_move_iterator(std::get<Is>(payloads1).end()))),
       ...);
    }(std::make_index_sequence<sizeof...(As)>());
  } else if(r < n2) {
    if(!resultKeys.empty() && resultKeys.back() == keys2[r]) {
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).back() =
              aggregators(std::get<Is>(resultValueVectors).back(), std::get<Is>(payloads2)[r])),
         ...);
      }(std::make_index_sequence<sizeof...(As)>());
      r++;
    }
    resultKeys.insert(resultKeys.end(), std::make_move_iterator(keys2.begin() + r),
                      std::make_move_iterator(keys2.end()));
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((std::get<Is>(resultValueVectors)
            .insert(std::get<Is>(resultValueVectors).end(),
                    std::make_move_iterator(std::get<Is>(payloads2).begin() + r),
                    std::make_move_iterator(std::get<Is>(payloads2).end()))),
       ...);
    }(std::make_index_sequence<sizeof...(As)>());
  }

  {
    std::lock_guard<std::mutex> lock(resultsMutex);
    results.push(std::make_tuple(resultKeys, resultKeys2, resultValueVectors));
    cv.notify_one();
  }
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments> groupByAdaptiveParallelMerge(
    std::condition_variable& cv, std::mutex& resultsMutex, int dop, bool secondKey,
    std::queue<AggregatedKeysAndPayload<K, As...>>& results, Aggregator<As>... aggregators) {

  auto& threadPool = ThreadPool::getInstance();

  for(int i = 0; i < dop - 1; ++i) {
    AggregatedKeysAndPayload<K, As...> mergeInput1;
    AggregatedKeysAndPayload<K, As...> mergeInput2;
    {
      std::unique_lock<std::mutex> lock(resultsMutex);
      cv.wait(lock, [&results] { return results.size() >= 2; });
      mergeInput1 = std::move(results.front());
      results.pop();
      mergeInput2 = std::move(results.front());
      results.pop();
    }
    threadPool.enqueue([&resultsMutex, &cv, mergeInput1 = std::move(mergeInput1),
                        mergeInput2 = std::move(mergeInput2), &results, aggregators...]() mutable {
      groupByAdaptiveParallelPerformMerge<K, As...>(resultsMutex, cv, std::move(mergeInput1),
                                                    std::move(mergeInput2), results,
                                                    aggregators...);
    });
  }

  std::unique_lock<std::mutex> lock(resultsMutex);
  cv.wait(lock, [&results] { return results.size() == 1; });

  AggregatedKeysAndPayload<K, As...> resultVectors = std::move(results.front());
  results.pop();

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);

  if(!secondKey) {
    result.emplace_back(Span<K>(std::move(std::get<0>(resultVectors))));
  } else {
    std::vector<K> resultKeys1 = std::move(std::get<0>(resultVectors));
    std::vector<K> resultKeys2 = std::move(std::get<1>(resultVectors));
    resultKeys2.reserve(resultKeys1.size());
    for(size_t i = 0; i < resultKeys1.size(); ++i) {
      resultKeys2.push_back(resultKeys1[i] >> FIRST_KEY_BITS);
      resultKeys1[i] = resultKeys1[i] & FIRST_KEY_MASK;
    }
    result.emplace_back(Span<K>(std::move(resultKeys1)));
    result.emplace_back(Span<K>(std::move(resultKeys2)));
  }

  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<As>(std::move(std::get<Is>(std::get<2>(resultVectors))))), ...);
  }(std::make_index_sequence<sizeof...(As)>());

  return result;
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupByAdaptiveParallel(int dop, int cardinality, bool secondKey,
                        ExpressionSpanArguments&& keySpans1, ExpressionSpanArguments&& keySpans2,
                        std::vector<Span<As>>&&... typedAggCols, Aggregator<As>... aggregators) {

  auto& threadPool = ThreadPool::getInstance();

  int n = 0;
  std::vector<int> spanSizes;
  spanSizes.reserve(keySpans1.size());
  for(const auto& untypedSpan : keySpans1) {
    auto& keySpan1 = std::get<Span<K>>(untypedSpan);
    n += keySpan1.size();
    spanSizes.push_back(keySpan1.size());
  }

  dop = std::min(dop, n);
  std::mutex resultsMutex;
  std::queue<AggregatedKeysAndPayload<K, As...>> results;
  std::condition_variable cv;

  int tuplesPerThreadBaseline = n / dop;
  int remainingTuples = n % dop;
  int outerIndex = 0;
  int innerIndex = 0;
  int tuplesPerThread;

  for(auto taskNum = 0; taskNum < dop; ++taskNum) {
    tuplesPerThread = tuplesPerThreadBaseline + (taskNum < remainingTuples);

    threadPool.enqueue([dop, &results, &resultsMutex, &cv, &spanSizes, tuplesPerThread, outerIndex,
                        innerIndex, cardinality, secondKey, &keySpans1, &keySpans2,
                        &typedAggCols..., aggregators...] {
      K largestKey = std::numeric_limits<K>::lowest();
      auto aggregatedKeysAndPayload = groupByAdaptive<K, As...>(
          dop, spanSizes, tuplesPerThread, outerIndex, innerIndex, cardinality, secondKey, false,
          largestKey, keySpans1, keySpans2, typedAggCols..., aggregators...);

      if(largestKey > 0) { // We only used hash-based Group, so need to sort result
        std::tuple<As*...> payloadPtrs = [&]<size_t... Is>(std::index_sequence<Is...>) {
          return std::make_tuple(std::get<Is>(std::get<2>(aggregatedKeysAndPayload)).data()...);
        }(std::make_index_sequence<sizeof...(As)>());
        sortByKey<K, As...>(std::get<0>(aggregatedKeysAndPayload).size(), largestKey,
                            std::get<0>(aggregatedKeysAndPayload).data(), payloadPtrs);
      }

      {
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.push(std::move(aggregatedKeysAndPayload));
        cv.notify_one();
      }
    });

    // Update outer and inner indexes to the start of the next thread
    while(tuplesPerThread && (spanSizes[outerIndex] - innerIndex) <= tuplesPerThread) {
      tuplesPerThread -= (spanSizes[outerIndex] - innerIndex);
      ++outerIndex;
      innerIndex = 0;
    }
    if(tuplesPerThread > 0) {
      innerIndex += tuplesPerThread;
    }
  }

  return groupByAdaptiveParallelMerge<K, As...>(cv, resultsMutex, dop, secondKey, results,
                                                aggregators...);
}

/*********************************** ENTRY FUNCTION ************************************/

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
group(Group implementation, int dop, int numKeys, ExpressionSpanArguments&& keySpans1,
      ExpressionSpanArguments&& keySpans2, std::vector<Span<As>>&&... typedAggCols,
      Aggregator<As>... aggregators) {
  assert(numKeys >= 0 && numKeys <= 2);
  int cardinality = getGroupResultCardinality();
  if(numKeys == 0) {
    return groupNoKeys<As...>(std::move(typedAggCols)..., aggregators...);
  }
#ifdef USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1
  if(implementation == Group::GroupAdaptiveParallel && dop == 1) {
    return groupByAdaptive<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                     std::move(keySpans2), std::move(typedAggCols)...,
                                     aggregators...);
  }
#endif
  switch(implementation) {
  case Group::Hash:
    assert(dop == 1);
    return groupByHash<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                 std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
  case Group::Sort:
    assert(dop == 1);
    return groupBySort<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                 std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
  case Group::GroupAdaptive:
    assert(dop == 1);
    return groupByAdaptive<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                     std::move(keySpans2), std::move(typedAggCols)...,
                                     aggregators...);
  case Group::GroupAdaptiveParallel:
    assert(adaptive::config::nonVectorizedDOP >= dop); // Will have a deadlock otherwise
    return groupByAdaptiveParallel<K, As...>(dop, cardinality, numKeys == 2, std::move(keySpans1),
                                             std::move(keySpans2), std::move(typedAggCols)...,
                                             aggregators...);
  default:
    throw std::runtime_error("Invalid selection of 'Group' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP