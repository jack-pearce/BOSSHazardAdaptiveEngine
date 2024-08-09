#ifndef BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>

#include "HazardAdaptiveEngine.hpp"
#include "constants/machineConstants.hpp"
#include "lazy_hash_map/robin_map.h"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/utilities.hpp"

#define ADAPTIVITY_OUTPUT
// #define DEBUG
// #define FAST_MERGE_DEBUG
#define USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1

// NOLINTBEGIN(readability-function-cognitive-complexity)

namespace adaptive {

namespace groupConfig {
constexpr float PERCENT_INPUT_TO_TRACK = 0.001;             // 0.1%
constexpr float PERCENT_INPUT_IN_TRANSIENT_CHECK = 0.00005; // 0.005%
constexpr int TUPLES_IN_CACHE_MISS_CHECK = 75 * 1000;       // 75,000
constexpr float PERCENT_INPUT_BETWEEN_HASHING = 0.25;       // 25%
constexpr int NUMBER_OF_TRANSIENT_CHECKS = 10;
constexpr int MIN_TUPLES_IN_TRANSIENT_CHECKS = 1000;

constexpr uint32_t FIRST_KEY_BITS = 8; // Bits for first key when there are two keys
constexpr int FIRST_KEY_MASK = static_cast<int>(1U << FIRST_KEY_BITS) - 1;
constexpr uint32_t BITS_PER_GROUP_RADIX_PASS = 8;
constexpr int DEFAULT_GROUP_RESULT_CARDINALITY = 100;
constexpr float HASHMAP_OVERALLOCATION_FACTOR = 2.5;
constexpr int MIN_HASHMAP_INITIAL_SIZE = 400'000;

constexpr int MIN_DUPLICATES_FOR_MULTI_THREADED_FALLBACK_TO_SORT = 10;
}

using config::minPartitionSize;
using namespace groupConfig;

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

template <typename T>
inline size_t findIndexFirstGreaterThanOrEqualTo(const std::vector<T>& sortedVector, T value) {
  size_t left = 0;
  size_t right = sortedVector.size();
  size_t result = -1;

  while(left < right) {
    size_t mid = left + (right - left) / 2;

    if(sortedVector[mid] >= value) {
      result = mid;
      right = mid;
    } else {
      left = mid + 1;
    }
  }

  if(result == static_cast<size_t>(-1)) {
    throw std::invalid_argument("No value in vector larger than requested value");
  }
  return result;
}

inline int getGroupResultCardinality() {
  auto getEnvironmentVariableNumber = [](const char* name, int& result) -> void {
    char* envVarStr = std::getenv(name);
    if(envVarStr != nullptr) {
      int value = std::atoi(envVarStr);
#ifdef DEBUG
      std::cout << "Read '" << name << "' environment variable value of: " << value << std::endl;
#endif
      result = value;
    } else {
#ifdef DEBUG
      std::cout << "Could not read '" << name << "' environment variable " << std::endl;
#endif
    }
  };

  int thisCardinality = DEFAULT_GROUP_RESULT_CARDINALITY;
  getEnvironmentVariableNumber("GROUP_RESULT_CARDINALITY", thisCardinality);
  int nextCardinality = -1;
  getEnvironmentVariableNumber("NEXT_GROUP_RESULT_CARDINALITY", nextCardinality);

  // Swap environment variables so that the correct value is picked up next i.e. repeatedly
  // swapping between the two values for repeated execution of a query with max two GROUP ops
  if(nextCardinality > 0) {
    std::string thisValue = std::to_string(nextCardinality);
    std::string nextValue = std::to_string(thisCardinality);
    setenv("GROUP_RESULT_CARDINALITY", thisValue.c_str(), 1);
    setenv("NEXT_GROUP_RESULT_CARDINALITY", nextValue.c_str(), 1);
  }

  return thisCardinality;
}

template <typename... As>
std::vector<ExpressionSpanArguments> groupNoKeys(std::vector<Span<As>>&&... typedAggCols,
                                                 Aggregator<As>... aggregators) {

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

  size_t spanNumber = 0;
  std::tuple<As...> resultValues = [&] {
    for(; spanNumber < sizes.size(); ++spanNumber) {
      if(sizes[spanNumber] > 0) {
        return std::make_tuple(typedAggCols[spanNumber][0]...);
      }
    }
    throw std::runtime_error("Input to 'groupNoKeys' function contains zero elements");
  }();

  for(size_t i = 1; i < sizes[spanNumber]; ++i) {
    resultValues = std::apply(
        [&](auto&&... args) {
          return std::make_tuple(aggregators(args, typedAggCols[spanNumber][i])...);
        },
        std::move(resultValues));
  }

  for(size_t i = spanNumber + 1; i < sizes.size(); ++i) {
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
  }
  (std::make_index_sequence<sizeof...(As)>());

  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(valueVectors)))), ...);
  }
  (std::make_index_sequence<sizeof...(As)>());

  return result;
}

template <typename K, typename... As>
void sortByKeyAux(size_t start, size_t end, K* keys, std::tuple<As*...> payloads, K* keysBuffer,
                  std::tuple<As*...> payloadBuffers, std::vector<int>& buckets,
                  uint32_t msbPosition, bool copyRequired) {
  size_t i = 0;
  uint32_t radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
  uint32_t shifts = msbPosition - radixBits;
  size_t numBuckets = 1U << radixBits;
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
    }
    (std::make_index_sequence<sizeof...(As)>());
  }

  std::fill(buckets.begin(), buckets.end(), 0);
  msbPosition -= static_cast<int>(radixBits);

  if(msbPosition == 0) {
    if(copyRequired) {
      std::memcpy(start + keys, start + keysBuffer, (end - start) * sizeof(K));
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::memcpy(start + std::get<Is>(payloads), start + std::get<Is>(payloadBuffers),
                      (end - start) * sizeof(std::tuple_element_t<Is, std::tuple<As...>>))),
         ...);
      }
      (std::make_index_sequence<sizeof...(As)>());
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
  uint32_t msbPosition = 0;
  while(largestKey != 0) {
    largestKey >>= 1;
    msbPosition++;
  }

  std::vector<int> buckets(1 + (1U << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n); // NOLINT
  auto* keysBuffer = keysBufferPtr.get();
  std::tuple<std::unique_ptr<As[]>...> payloadBufferPtrs = // NOLINT
      [&]<size_t... Is>(std::index_sequence<Is...>) {      // NOLINT
    return std::make_tuple(
        std::make_unique_for_overwrite<std::tuple_element_t<Is, std::tuple<As...>>[]>( // NOLINT
            n)...);
  }
  (std::make_index_sequence<sizeof...(As)>());
  std::tuple<As*...> payloadBuffers = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    return std::make_tuple(std::get<Is>(payloadBufferPtrs).get()...);
  }
  (std::make_index_sequence<sizeof...(As)>());

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
  int initialSize =
      std::max(static_cast<int>(HASHMAP_OVERALLOCATION_FACTOR * static_cast<float>(cardinality)),
               MIN_HASHMAP_INITIAL_SIZE);
  int index = 0;

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
    }
    (std::make_index_sequence<sizeof...(As)>());
  }

  result.emplace_back(Span<std::remove_cv_t<K>>(std::move(keys)));
  if(secondKey) {
    result.emplace_back(Span<std::remove_cv_t<K>>(std::move(*keys2)));
  }
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(valueVectors)))), ...);
  }
  (std::make_index_sequence<sizeof...(As)>());

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
    }
    (std::make_index_sequence<sizeof...(As)>());
    ++i;
    while(keys[i] == key) {
      tuple = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        return std::make_tuple(aggregators(std::get<Is>(tuple), std::get<Is>(payloads[i]))...);
      }
      (std::make_index_sequence<sizeof...(As)>());
      ++i;
    }
    addKeys(key);
    [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(tuple))), ...);
    }
    (std::make_index_sequence<sizeof...(As)>());
  }
}

template <typename K, typename... As>
inline void groupBySortFinalPassAndAggSingleThread(
    size_t n, K* keys, std::tuple<As...>* payloads, std::vector<std::remove_cv_t<K>>& resultKeys,
    std::vector<std::remove_cv_t<K>>& resultKeys2,
    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors, bool secondKey,
    bool splitKeysInResult, uint32_t msbPosition, Aggregator<As>... aggregators) {

  static bool bucketEntryPresent[1U << BITS_PER_GROUP_RADIX_PASS];       // NOLINT
  static std::tuple<As...> payloadAggs[1U << BITS_PER_GROUP_RADIX_PASS]; // NOLINT
  std::fill(std::begin(bucketEntryPresent), std::end(bucketEntryPresent), false);

  auto addKeys = [&resultKeys, &resultKeys2, secondKey, splitKeysInResult](auto key) mutable {
    if(secondKey && splitKeysInResult) {
      resultKeys.push_back(key & FIRST_KEY_MASK);
      resultKeys2.push_back(key >> FIRST_KEY_BITS);
    } else {
      resultKeys.push_back(key);
    }
  };

  size_t i = 0;
  uint32_t radixBits = msbPosition;
  size_t numBuckets = 1U << radixBits;
  uint32_t mask = static_cast<int>(numBuckets) - 1;

  for(i = 0; i < n; i++) {
    auto keyLowerBits = keys[i] & mask;
    payloadAggs[keyLowerBits] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      if(bucketEntryPresent[keyLowerBits]) {
        return std::make_tuple(
            aggregators(std::get<Is>(payloadAggs[keyLowerBits]), std::get<Is>(payloads[i]))...);
      }
      return std::make_tuple(std::get<Is>(payloads[i])...);
    }
    (std::make_index_sequence<sizeof...(As)>());
    bucketEntryPresent[keyLowerBits] = true;
  }

  uint32_t valuePrefix = keys[0] & ~mask;

  for(i = 0; i < numBuckets; i++) {
    if(bucketEntryPresent[i]) { // NOLINT
      addKeys(valuePrefix | i);
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloadAggs[i]))), ...);
      }
      (std::make_index_sequence<sizeof...(As)>());
    }
  }
}

template <typename K, typename... As>
inline void groupBySortFinalPassAndAggMultiThread(
    size_t n, K* keys, std::tuple<As...>* payloads, std::vector<std::remove_cv_t<K>>& resultKeys,
    std::vector<std::remove_cv_t<K>>& resultKeys2,
    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors, bool secondKey,
    bool splitKeysInResult, uint32_t msbPosition, Aggregator<As>... aggregators) {

  static thread_local bool bucketEntryPresent[1U << BITS_PER_GROUP_RADIX_PASS];       // NOLINT
  static thread_local std::tuple<As...> payloadAggs[1U << BITS_PER_GROUP_RADIX_PASS]; // NOLINT
  std::fill(std::begin(bucketEntryPresent), std::end(bucketEntryPresent), false);

  auto addKeys = [&resultKeys, &resultKeys2, secondKey, splitKeysInResult](auto key) mutable {
    if(secondKey && splitKeysInResult) {
      resultKeys.push_back(key & FIRST_KEY_MASK);
      resultKeys2.push_back(key >> FIRST_KEY_BITS);
    } else {
      resultKeys.push_back(key);
    }
  };

  size_t i = 0;
  uint32_t radixBits = msbPosition;
  size_t numBuckets = 1U << radixBits;
  uint32_t mask = static_cast<int>(numBuckets) - 1;

  for(i = 0; i < n; i++) {
    auto keyLowerBits = keys[i] & mask;
    payloadAggs[keyLowerBits] = [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      if(bucketEntryPresent[keyLowerBits]) {
        return std::make_tuple(
            aggregators(std::get<Is>(payloadAggs[keyLowerBits]), std::get<Is>(payloads[i]))...);
      }
      return std::make_tuple(std::get<Is>(payloads[i])...);
    }
    (std::make_index_sequence<sizeof...(As)>());
    bucketEntryPresent[keyLowerBits] = true;
  }

  uint32_t valuePrefix = keys[0] & ~mask;

  for(i = 0; i < numBuckets; i++) {
    if(bucketEntryPresent[i]) { // NOLINT
      addKeys(valuePrefix | i);
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloadAggs[i]))), ...);
      }
      (std::make_index_sequence<sizeof...(As)>());
    }
  }
}

// This and the prior two functions is only to remove the thread_local keyword when single-threaded
// since this was observed to have an overhead when profiled
template <typename K, typename... As>
void groupBySortFinalPassAndAgg(
    int dop, size_t n, K* keys, std::tuple<As...>* payloads,
    std::vector<std::remove_cv_t<K>>& resultKeys, std::vector<std::remove_cv_t<K>>& resultKeys2,
    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors, bool secondKey,
    bool splitKeysInResult, uint32_t msbPosition, Aggregator<As>... aggregators) {
  if(dop == 1) {
    groupBySortFinalPassAndAggSingleThread(n, keys, payloads, resultKeys, resultKeys2,
                                           resultValueVectors, secondKey, splitKeysInResult,
                                           msbPosition, aggregators...);
  } else {
    groupBySortFinalPassAndAggMultiThread(n, keys, payloads, resultKeys, resultKeys2,
                                          resultValueVectors, secondKey, splitKeysInResult,
                                          msbPosition, aggregators...);
  }
}

template <typename K, typename... As>
void groupBySortAux(int dop, size_t n, std::vector<int>& buckets, K* keys,
                    std::tuple<As...>* payloads, K* keysBuffer, std::tuple<As...>* payloadsBuffer,
                    std::vector<std::remove_cv_t<K>>& resultKeys,
                    std::vector<std::remove_cv_t<K>>& resultKeys2,
                    std::tuple<std::vector<std::remove_cv_t<As>>...>& resultValueVectors,
                    bool secondKey, bool splitKeysInResult, uint32_t msbPosition,
                    Aggregator<As>... aggregators) {
  size_t i = 0;
  uint32_t radixBits = BITS_PER_GROUP_RADIX_PASS;
  size_t numBuckets = 1U << radixBits;
  uint32_t mask = static_cast<int>(numBuckets) - 1;
  uint32_t shifts = msbPosition - radixBits;

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
            dop, partitions[i] - prevPartitionEnd, prevPartitionEnd + keysBuffer,
            prevPartitionEnd + payloadsBuffer, resultKeys, resultKeys2, resultValueVectors,
            secondKey, splitKeysInResult, msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  } else {
    int prevPartitionEnd = 0;
    for(i = 0; i < numBuckets; i++) {
      if(partitions[i] != prevPartitionEnd) {
        groupBySortAux<K, As...>(dop, partitions[i] - prevPartitionEnd, buckets,
                                 prevPartitionEnd + keysBuffer, prevPartitionEnd + payloadsBuffer,
                                 prevPartitionEnd + keys, prevPartitionEnd + payloads, resultKeys,
                                 resultKeys2, resultValueVectors, secondKey, splitKeysInResult,
                                 msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  }
}

// This class acts as the Dispatcher
template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupBySort(int dop, int cardinality, bool secondKey, ExpressionSpanArguments&& keySpans1,
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
  uint32_t msbPosition = 0;
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

  std::vector<int> buckets(1 + (1U << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysPtr = std::make_unique_for_overwrite<K[]>(n); // NOLINT
  auto* keys = keysPtr.get();
  auto payloadsPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n); // NOLINT
  auto* payloads = payloadsPtr.get();
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n); // NOLINT
  auto* keysBuffer = keysBufferPtr.get();
  auto payloadsBufferPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n); // NOLINT
  auto* payloadsBuffer = payloadsBufferPtr.get();

  size_t i = 0;
  uint32_t radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
  size_t numBuckets = 1U << radixBits;
  uint32_t mask = static_cast<int>(numBuckets) - 1;
  uint32_t shifts = msbPosition - radixBits;

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
        groupBySortFinalPassAndAgg<K, As...>(dop, partitions[i] - prevPartitionEnd,
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
        groupBySortAux<K, As...>(dop, partitions[i] - prevPartitionEnd, buckets,
                                 prevPartitionEnd + keys, prevPartitionEnd + payloads,
                                 prevPartitionEnd + keysBuffer, prevPartitionEnd + payloadsBuffer,
                                 resultKeys, resultKeys2, resultValueVectors, secondKey, true,
                                 msbPosition, aggregators...);
      }
      prevPartitionEnd = partitions[i];
    }
  }

  result.emplace_back(Span<std::remove_cv_t<K>>(std::move(resultKeys)));
  if(secondKey) {
    result.emplace_back(Span<std::remove_cv_t<K>>(std::move(resultKeys2)));
  }
  [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
    (result.emplace_back(Span<std::remove_cv_t<As>>(std::move(std::get<Is>(resultValueVectors)))),
     ...);
  }
  (std::make_index_sequence<sizeof...(As)>());

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
           (measuredPageFaultDecreaseRatePerTuple > 0 ||
            std::abs(measuredPageFaultDecreaseRatePerTuple) < pageFaultDecreaseRatePerTuple);
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
groupByAdaptiveAuxSort(int dop, HA_tsl::robin_map<K, std::tuple<As...>>& map, K& largestKey,
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

  size_t i = 0;
  uint32_t msbPosition = 0;

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

  std::vector<int> buckets(1 + (1U << BITS_PER_GROUP_RADIX_PASS), 0);
  auto keysPtr = std::make_unique_for_overwrite<K[]>(n); // NOLINT
  auto* keys = keysPtr.get();
  auto payloadsPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n); // NOLINT
  auto* payloads = payloadsPtr.get();
  auto keysBufferPtr = std::make_unique_for_overwrite<K[]>(n); // NOLINT
  auto* keysBuffer = keysBufferPtr.get();
  auto payloadsBufferPtr = std::make_unique_for_overwrite<std::tuple<As...>[]>(n); // NOLINT
  auto* payloadsBuffer = payloadsBufferPtr.get();

  uint32_t radixBits = std::min(msbPosition, BITS_PER_GROUP_RADIX_PASS);
  size_t numBuckets = 1U << radixBits;
  uint32_t mask = numBuckets - 1;
  uint32_t shifts = msbPosition - radixBits;

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
        }
        (std::make_index_sequence<sizeof...(As)>());
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
        }
        (std::make_index_sequence<sizeof...(As)>());
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
        groupBySortFinalPassAndAgg<K, As...>(dop, partitions[i] - prevPartitionEnd,
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
        groupBySortAux<K, As...>(dop, partitions[i] - prevPartitionEnd, buckets,
                                 prevPartitionEnd + keys, prevPartitionEnd + payloads,
                                 prevPartitionEnd + keysBuffer, prevPartitionEnd + payloadsBuffer,
                                 resultKeys, resultKeys2, resultValueVectors, secondKey,
                                 splitKeysInResult, msbPosition, aggregators...);
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

  // Fall back to sort straight away
  if(dop > 1 && (n / cardinality) < MIN_DUPLICATES_FOR_MULTI_THREADED_FALLBACK_TO_SORT) {
#ifdef FAST_MERGE_DEBUG
    std::cout << "Falling back to sort-based implementation immediately" << std::endl;
#endif
    // Empty map related variables
    HA_tsl::robin_map<K, std::tuple<As...>> map(0);
    int entriesInMapToTrack = 0;
    std::pair<K, std::tuple<As...>>** mapPtrs = nullptr;

    std::vector<Section<K, As...>> sectionsToBeSorted;
    int elements = n;

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

    int tuplesToProcess = n;
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

    return groupByAdaptiveAuxSort<K, As...>(dop, map, largestKey, mapPtrs, entriesInMapToTrack,
                                            elements, cardinality, sectionsToBeSorted, secondKey,
                                            splitKeysInResult, aggregators...);
  }

  int tuplesInTransientCheck =
      static_cast<int>(PERCENT_INPUT_IN_TRANSIENT_CHECK * static_cast<float>(n));
  int tuplesPerTransientCheckReading =
      static_cast<int>(std::ceil(tuplesInTransientCheck / NUMBER_OF_TRANSIENT_CHECKS));
  tuplesInTransientCheck = tuplesPerTransientCheckReading * NUMBER_OF_TRANSIENT_CHECKS;
  std::vector<int> tuplesPerReading;
  if(tuplesInTransientCheck < MIN_TUPLES_IN_TRANSIENT_CHECKS || tuplesInTransientCheck > n) {
    tuplesInTransientCheck = 0;
  } else {
    tuplesPerReading.reserve(NUMBER_OF_TRANSIENT_CHECKS);
    for(int i = 1; i < (NUMBER_OF_TRANSIENT_CHECKS + 1); i++) {
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

  int initialSize =
      std::max(static_cast<int>(HASHMAP_OVERALLOCATION_FACTOR * static_cast<float>(cardinality)),
               MIN_HASHMAP_INITIAL_SIZE);
  HA_tsl::robin_map<K, std::tuple<As...>> map(initialSize);

  int entriesInMapToTrack = static_cast<int>(PERCENT_INPUT_TO_TRACK * static_cast<float>(n));
  int mapPtrsSize = 0;
  auto mapPtrsRaii = std::make_unique_for_overwrite<std::pair<K, std::tuple<As...>>*[]>( // NOLINT
      entriesInMapToTrack);
  auto* mapPtrs = mapPtrsRaii.get();

  auto& eventSet = getThreadEventSet();
  long_long* baseEventPtr = eventSet.getCounterDiffsPtr();

  auto& constants = MachineConstants::getInstance();
  constexpr int numBytes = static_cast<int>(std::min((sizeof(K) + ... + sizeof(As)), 48UL));
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
  int tuplesToProcess, tuplesInCheck;                       // NOLINT
  double pageFaultDecreaseRatePerTuple, pageFaultsPerTuple; // NOLINT

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
      pageFaults.reserve(NUMBER_OF_TRANSIENT_CHECKS);
      for(int i = 0; i < NUMBER_OF_TRANSIENT_CHECKS; i++) {
        tuplesToProcess = tuplesInTransientCheck / NUMBER_OF_TRANSIENT_CHECKS;
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
      pageFaultDecreaseRatePerTuple = linearRegressionSlope(tuplesPerReading, pageFaults);
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
      }
      (std::make_index_sequence<sizeof...(As)>());
    }

    return std::make_tuple(resultKeys, resultKeys2, resultValueVectors);
  }

  elements += map.size();
  return groupByAdaptiveAuxSort<K, As...>(dop, map, largestKey, mapPtrs, entriesInMapToTrack,
                                          elements, cardinality, sectionsToBeSorted, secondKey,
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

  auto [keys1, keys2, payloads] = std::move(resultVectors);
  auto keysPtr1 = std::make_shared<std::vector<K>>(std::move(keys1));
  auto keysPtr2 = std::make_shared<std::vector<K>>(std::move(keys2));
  auto payloadsPtrs = std::apply(
      [&](auto&&... vecs) {
        return std::make_tuple(std::make_shared<std::vector<As>>(std::move(vecs))...);
      },
      std::move(payloads));

  size_t numResultSpans = (keysPtr1->size() + minPartitionSize - 1) / minPartitionSize;

  ExpressionSpanArguments key1ResultSpans;
  key1ResultSpans.reserve(numResultSpans);
  size_t spanStart = 0;
  size_t spanSize = minPartitionSize;
  for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
    key1ResultSpans.emplace_back(
        Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
    spanStart += spanSize;
  }
  spanSize = keysPtr1->size() - spanStart;
  if(spanSize > 0) {
    key1ResultSpans.emplace_back(
        Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
  }
  result.push_back(std::move(key1ResultSpans));

  if(secondKey) {
    ExpressionSpanArguments key2ResultSpans;
    key2ResultSpans.reserve(numResultSpans);
    size_t spanStart = 0;
    size_t spanSize = minPartitionSize;
    for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
      key2ResultSpans.emplace_back(
          Span<K>(keysPtr2->data() + spanStart, spanSize, [ptr = keysPtr2]() {}));
      spanStart += spanSize;
    }
    spanSize = keysPtr2->size() - spanStart;
    if(spanSize > 0) {
      key2ResultSpans.emplace_back(
          Span<K>(keysPtr2->data() + spanStart, spanSize, [ptr = keysPtr2]() {}));
    }
    result.push_back(std::move(key2ResultSpans));
  }

  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (
        [&]() {
          ExpressionSpanArguments payloadSpans;
          payloadSpans.reserve(numResultSpans);
          auto payloadPtr = std::get<Is>(payloadsPtrs);
          size_t spanStart = 0;
          size_t spanSize = minPartitionSize;
          for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
            payloadSpans.emplace_back(
                Span<As>(payloadPtr->data() + spanStart, spanSize, [ptr = payloadPtr]() {}));
            spanStart += spanSize;
          }
          spanSize = payloadPtr->size() - spanStart;
          if(spanSize > 0) {
            payloadSpans.emplace_back(
                Span<As>(payloadPtr->data() + spanStart, spanSize, [ptr = payloadPtr]() {}));
          }
          result.push_back(std::move(payloadSpans));
        }(),
        ...);
  }
  (std::make_index_sequence<sizeof...(As)>());

  return result;
}

/************************************ MULTI-THREADED ***********************************/

template <typename K, typename... As>
void groupByAdaptiveParallelPerformMerge(std::mutex& resultsMutex, std::condition_variable& cv,
                                         AggregatedKeysAndPayload<K, As...>&& mergeInput1,
                                         AggregatedKeysAndPayload<K, As...>&& mergeInput2,
                                         std::queue<AggregatedKeysAndPayload<K, As...>>& results,
                                         Aggregator<As>... aggregators) {
  using Keys = std::vector<K>;
  using Payloads = std::tuple<std::vector<As>...>;

  Keys keys1 = std::move(std::get<0>(mergeInput1));
  Keys keys2 = std::move(std::get<0>(mergeInput2));
  Payloads payloads1 = std::move(std::get<2>(mergeInput1));
  Payloads payloads2 = std::move(std::get<2>(mergeInput2));
  size_t index1 = 0;
  size_t index2 = 0;

  size_t n1 = keys1.size();
  size_t n2 = keys2.size();
  size_t maxOutputSize = n1 + n2;

  Keys resultKeys;
  resultKeys.reserve(maxOutputSize);
  Keys resultKeys2;
  Payloads resultValueVectors;
  std::apply([&](auto&&... vec) { (vec.reserve(maxOutputSize), ...); }, resultValueVectors);

  auto pushResultsToQueue = [&]() {
    std::lock_guard<std::mutex> lock(resultsMutex);
    results.push(std::make_tuple(resultKeys, resultKeys2, resultValueVectors));
    cv.notify_one();
  };

  auto copyIntoResults = [&](Keys& keys, Payloads& payloads, size_t startElement,
                             size_t numElements) -> void {
    size_t currentResultElements = resultKeys.size();
    resultKeys.resize(resultKeys.size() + numElements);
    std::apply([&](auto&&... vec) { (vec.resize(vec.size() + numElements), ...); },
               resultValueVectors);
    memcpy(&resultKeys[currentResultElements], &keys[startElement], numElements * sizeof(K));
    [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
      ((memcpy(&std::get<Is>(resultValueVectors)[currentResultElements],
               &std::get<Is>(payloads)[startElement],
               numElements * sizeof(typename std::tuple_element<Is, Payloads>::type::value_type))),
       ...);
    }
    (std::make_index_sequence<sizeof...(As)>());
  };

  if(n1 == 0) {
    copyIntoResults(keys1, payloads1, 0, keys1.size());
    pushResultsToQueue();
    return;
  }
  if(n2 == 0) {
    copyIntoResults(keys2, payloads2, 0, keys2.size());
    pushResultsToQueue();
    return;
  }

  K minKey1 = keys1[0];
  K minKey2 = keys2[0];
  K maxKey1 = keys1.back();
  K maxKey2 = keys2.back();

  {
    bool minKey1LessThanMinKey2 = minKey1 < minKey2;
    Keys& minKeys = minKey1LessThanMinKey2 ? keys1 : keys2;
    Keys& otherKeys = minKey1LessThanMinKey2 ? keys2 : keys1;
    Payloads& minPayloads = minKey1LessThanMinKey2 ? payloads1 : payloads2;
    Payloads& otherPayloads = minKey1LessThanMinKey2 ? payloads2 : payloads1;

    if(maxKey1 < minKey2 || maxKey2 < minKey1) { // no overlap (no merge required)
      copyIntoResults(minKeys, minPayloads, 0, minKeys.size());
      copyIntoResults(otherKeys, otherPayloads, 0, otherKeys.size());
      pushResultsToQueue();
      return;
    }

    // Binary search to find point at which overlap starts for minKeys
    size_t numNotOverlapping = findIndexFirstGreaterThanOrEqualTo<K>(minKeys, otherKeys[0]);

    // copy any non-overlapping values into results
    copyIntoResults(minKeys, minPayloads, 0, numNotOverlapping);
    size_t& minIndex = minKey1LessThanMinKey2 ? index1 : index2;
    minIndex += numNotOverlapping;
  }

  {
    bool maxKey1GreaterThanMaxKey2 = maxKey1 > maxKey2;
    Keys& maxKeys = maxKey1GreaterThanMaxKey2 ? keys1 : keys2;
    Keys& otherKeys = maxKey1GreaterThanMaxKey2 ? keys2 : keys1;
    Payloads& maxPayloads = maxKey1GreaterThanMaxKey2 ? payloads1 : payloads2;
    Payloads& otherPayloads = maxKey1GreaterThanMaxKey2 ? payloads2 : payloads1;
    size_t& indexMax = maxKey1GreaterThanMaxKey2 ? index1 : index2;
    size_t& indexOther = maxKey1GreaterThanMaxKey2 ? index2 : index1;

    // Merge overlapping region (i.e. all remaining 'otherKeys')
    while(indexOther < otherKeys.size()) {
      if(otherKeys[indexOther] < maxKeys[indexMax]) {
        resultKeys.push_back(otherKeys[indexOther]);
        [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
          ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(otherPayloads)[indexOther])),
           ...);
        }
        (std::make_index_sequence<sizeof...(As)>());
        indexOther++;
      } else if(maxKeys[indexMax] < otherKeys[indexOther]) {
        resultKeys.push_back(maxKeys[indexMax]);
        [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
          ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(maxPayloads)[indexMax])), ...);
        }
        (std::make_index_sequence<sizeof...(As)>());
        indexMax++;
      } else {
        resultKeys.push_back(otherKeys[indexOther]);
        [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
          ((std::get<Is>(resultValueVectors)
                .push_back(aggregators(std::get<Is>(otherPayloads)[indexOther],
                                       std::get<Is>(maxPayloads)[indexMax]))),
           ...);
        }
        (std::make_index_sequence<sizeof...(As)>());
        indexOther++;
        indexMax++;
      }
    }

    // copy any remaining values into results
    copyIntoResults(maxKeys, maxPayloads, indexMax, maxKeys.size() - indexMax);
  }

  pushResultsToQueue();
}

struct Interval {
  size_t start;
  size_t end;
  size_t taskNum;
};

// Intervals must be sorted by start and then by end
inline bool containsOnlySequentialOverlappingIntervals(const std::vector<Interval>& intervals) {
  for(size_t i = 0; i < intervals.size() - 1; ++i) {
    const Interval& current = intervals[i];
    const Interval& next = intervals[i + 1];

    if(current.end >= next.end) {
      return false; // Fully overlapping interval
    }

    if(i < intervals.size() - 2) {
      const Interval& nextNext = intervals[i + 2];
      if(current.end > next.start && current.end > nextNext.start) {
        return false; // More than two overlapping intervals
      }
    }
  }
#ifdef FAST_MERGE_DEBUG
  std::cout
      << "Intervals do only contain sequential overlapping intervals, will run fast parallel merge"
      << std::endl;
#endif
  return true;
}

// Intervals must be sorted by start and then by end
template <typename K, typename... As>
AggregatedKeysAndPayload<K, As...>
groupByAdaptiveFastParallelMerge(const std::vector<Interval>& intervals, std::mutex& resultsMutex,
                                 int dop, std::vector<AggregatedKeysAndPayload<K, As...>>& results,
                                 Aggregator<As>... aggregators) {

  auto& threadPool = ThreadPool::getInstance(dop);
  auto& synchroniser = Synchroniser::getInstance();

  using Keys = std::vector<K>;
  using Payloads = std::tuple<std::vector<As>...>;

  size_t maxResultSize = 0;
  for(const auto& intermediateResult : results) {
    maxResultSize += std::get<0>(intermediateResult).size();
  }

  Keys resultKeys(maxResultSize);
  Keys resultKeys2;
  Payloads resultValueVectors;
  std::apply([&](auto&&... vec) { (vec.resize(maxResultSize), ...); }, resultValueVectors);
  size_t resultsSize = 0;

  std::queue<AggregatedKeysAndPayload<K, As...>> overlappingResults;
  int totalTasks = 0;

  auto firstIndexOfValueGreater = [](const Keys& keys, K value) {
    for(size_t i = 0; i < keys.size(); i++) {
      if(keys[i] > value) {
        return i;
      }
    }
    throw std::runtime_error("Expected at least one value greater but none found");
  };

  auto lastIndexOfValueSmaller = [](const std::vector<K>& keys, K value) {
    for(size_t i = keys.size(); i > 0; i--) {
      if(keys[i - 1] < value) {
        return i - 1;
      }
    }
    throw std::runtime_error("Expected at least one value smaller but none found");
  };

  size_t startIndexInInterval = 0;
  size_t endIndexInInterval = 0;
  for(size_t i = 0; i < intervals.size(); ++i) { // Process interval and any overlap
    if(i + 1 < intervals.size() && intervals[i + 1].start <= intervals[i].end) { // overlap

      endIndexInInterval = lastIndexOfValueSmaller(std::get<0>(results[intervals[i].taskNum]),
                                                   intervals[i + 1].start);
      {
        Keys& keys = std::get<0>(results[intervals[i].taskNum]);
        Payloads& payloads = std::get<2>(results[intervals[i].taskNum]);
        size_t index = resultsSize;
        resultsSize += (endIndexInInterval + 1 - startIndexInInterval);
#ifdef FAST_MERGE_DEBUG
        std::cout << "Copy from task " << i << " index [" << startIndexInInterval << ", "
                  << endIndexInInterval << "]" << std::endl;
        std::cout << "Last value = " << keys[endIndexInInterval] << std::endl;
#endif
        totalTasks++;
        threadPool.enqueue([&synchroniser, &resultKeys, &resultValueVectors, &keys, &payloads,
                            startElement = startIndexInInterval,
                            numElements = (endIndexInInterval + 1 - startIndexInInterval),
                            index]() mutable {
          memcpy(&resultKeys[index], &keys[startElement], numElements * sizeof(K));
          [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
            ((memcpy(&std::get<Is>(resultValueVectors)[index],
                     &std::get<Is>(payloads)[startElement],
                     numElements *
                         sizeof(typename std::tuple_element<Is, Payloads>::type::value_type))),
             ...);
          }
          (std::make_index_sequence<sizeof...(As)>());
          synchroniser.taskComplete();
        });
      }

      startIndexInInterval = firstIndexOfValueGreater(
          std::get<0>(results[intervals[i + 1].taskNum]), intervals[i].end);
      {
        Keys& keys1 = std::get<0>(results[intervals[i].taskNum]);
        Payloads& payloads1 = std::get<2>(results[intervals[i].taskNum]);
        Keys& keys2 = std::get<0>(results[intervals[i + 1].taskNum]);
        Payloads& payloads2 = std::get<2>(results[intervals[i + 1].taskNum]);
#ifdef FAST_MERGE_DEBUG
        std::cout << "Merge task " << i << " index [" << endIndexInInterval + 1 << ", "
                  << keys1.size() << ") with "
                  << " task " << i + 1 << " index [" << 0 << ", " << startIndexInInterval << ")"
                  << std::endl;
        std::cout << "Task " << i << " first value = " << keys1[endIndexInInterval + 1]
                  << ", last value = " << keys1[keys1.size() - 1] << std::endl;
        std::cout << "Task " << i + 1 << " first value = " << keys2[0]
                  << ", last value = " << keys2[startIndexInInterval - 1] << std::endl;
#endif
        totalTasks++;
        threadPool.enqueue([&synchroniser, &overlappingResults, &resultsMutex, &keys1, &keys2,
                            &payloads1, &payloads2, startElement1 = endIndexInInterval + 1,
                            numElements1 = (keys1.size() - (endIndexInInterval + 1)),
                            startElement2 = 0, numElements2 = startIndexInInterval,
                            aggregators...]() mutable {
          size_t end1 = startElement1 + numElements1;
          size_t end2 = startElement2 + numElements2;
          size_t maxOutputSize = numElements1 + numElements2;

          std::vector<K> resultKeys;
          resultKeys.reserve(maxOutputSize);
          std::vector<K> resultKeys2;
          std::tuple<std::vector<As>...> resultValueVectors;
          std::apply([&](auto&&... vec) { (vec.reserve(maxOutputSize), ...); }, resultValueVectors);

          size_t l = startElement1;
          size_t r = startElement2;

          while(l < end1 && r < end2) {
            if(keys1[l] < keys2[r]) {
              resultKeys.push_back(keys1[l]);
              [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
                ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloads1)[l])), ...);
              }
              (std::make_index_sequence<sizeof...(As)>());
              l++;
            } else if(keys2[r] < keys1[l]) {
              resultKeys.push_back(keys2[r]);
              [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
                ((std::get<Is>(resultValueVectors).push_back(std::get<Is>(payloads2)[r])), ...);
              }
              (std::make_index_sequence<sizeof...(As)>());
              r++;
            } else {
              resultKeys.push_back(keys1[l]);
              [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
                ((std::get<Is>(resultValueVectors)
                      .push_back(
                          aggregators(std::get<Is>(payloads1)[l], std::get<Is>(payloads2)[r]))),
                 ...);
              }
              (std::make_index_sequence<sizeof...(As)>());
              l++;
              r++;
            }
          }

          if(l < end1) {
            if(!resultKeys.empty() && resultKeys.back() == keys1[l]) {
              [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
                ((std::get<Is>(resultValueVectors).back() = aggregators(
                      std::get<Is>(resultValueVectors).back(), std::get<Is>(payloads1)[l])),
                 ...);
              }
              (std::make_index_sequence<sizeof...(As)>());
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
            }
            (std::make_index_sequence<sizeof...(As)>());
          } else if(r < end2) {
            if(!resultKeys.empty() && resultKeys.back() == keys2[r]) {
              [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
                ((std::get<Is>(resultValueVectors).back() = aggregators(
                      std::get<Is>(resultValueVectors).back(), std::get<Is>(payloads2)[r])),
                 ...);
              }
              (std::make_index_sequence<sizeof...(As)>());
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
            }
            (std::make_index_sequence<sizeof...(As)>());
          }

          {
            std::lock_guard<std::mutex> lock(resultsMutex);
            overlappingResults.push(std::make_tuple(resultKeys, resultKeys2, resultValueVectors));
          }

          synchroniser.taskComplete();
        });
      }

    } else { // no overlap

      {
        Keys& keys = std::get<0>(results[intervals[i].taskNum]);
        Payloads& payloads = std::get<2>(results[intervals[i].taskNum]);
        size_t index = resultsSize;
        resultsSize += (keys.size() - startIndexInInterval);
#ifdef FAST_MERGE_DEBUG
        std::cout << "Copy from task " << i << " index [" << startIndexInInterval << ", "
                  << keys.size() << ")" << std::endl;
        std::cout << "First value = " << keys[startIndexInInterval] << std::endl;
#endif
        totalTasks++;
        threadPool.enqueue([&synchroniser, &resultKeys, &resultValueVectors, &keys, &payloads,
                            startElement = startIndexInInterval,
                            numElements = (keys.size() - startIndexInInterval), index]() mutable {
          memcpy(&resultKeys[index], &keys[startElement], numElements * sizeof(K));
          [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
            ((memcpy(&std::get<Is>(resultValueVectors)[index],
                     &std::get<Is>(payloads)[startElement],
                     numElements *
                         sizeof(typename std::tuple_element<Is, Payloads>::type::value_type))),
             ...);
          }
          (std::make_index_sequence<sizeof...(As)>());

          synchroniser.taskComplete();
        });
      }

      startIndexInInterval = 0;
    }
  }
  synchroniser.waitUntilComplete(totalTasks);

  totalTasks = 0;
  while(!overlappingResults.empty()) {
    AggregatedKeysAndPayload<K, As...> mergedResult = std::move(overlappingResults.front());
    overlappingResults.pop();
    Keys& keys = std::get<0>(mergedResult);
    Payloads& payloads = std::get<2>(mergedResult);
    size_t index = resultsSize;
    resultsSize += keys.size();
    totalTasks++;
    threadPool.enqueue([&synchroniser, &resultKeys, &resultValueVectors, &keys, &payloads,
                        numElements = keys.size(), index]() mutable {
      memcpy(&resultKeys[index], keys.data(), numElements * sizeof(K));
      [&]<size_t... Is>(std::index_sequence<Is...>) { // NOLINT
        ((memcpy(&std::get<Is>(resultValueVectors)[index], &std::get<Is>(payloads)[0],
                 numElements *
                     sizeof(typename std::tuple_element<Is, Payloads>::type::value_type))),
         ...);
      }
      (std::make_index_sequence<sizeof...(As)>());

      synchroniser.taskComplete();
    });
  }
  synchroniser.waitUntilComplete(totalTasks);
  resultKeys.resize(resultsSize);
  std::apply([&](auto&&... vec) { (vec.resize(resultsSize), ...); }, resultValueVectors);
  return {resultKeys, resultKeys2, resultValueVectors};
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments> groupByAdaptiveParallelMerge(
    std::condition_variable& cv, std::mutex& resultsMutex, int dop, bool secondKey,
    std::queue<AggregatedKeysAndPayload<K, As...>>& results, Aggregator<As>... aggregators) {
  std::vector<AggregatedKeysAndPayload<K, As...>> resultsVec;
  resultsVec.reserve(results.size());
  std::vector<Interval> intervals;
  intervals.reserve(results.size());
  size_t taskNum = 0;
  while(!results.empty()) {
    resultsVec.push_back(std::move(results.front()));
    results.pop();
    auto start = static_cast<size_t>(std::get<0>(resultsVec.back()).front());
    auto end = static_cast<size_t>(std::get<0>(resultsVec.back()).back());
    intervals.push_back({start, end, taskNum++});
  }

  // Sort intervals by start time, and by end time in case of tie
  std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
    if(a.start == b.start) {
      return a.end < b.end;
    }
    return a.start < b.start;
  });

  AggregatedKeysAndPayload<K, As...> resultVectors;
  if(containsOnlySequentialOverlappingIntervals(intervals)) {
    resultVectors = groupByAdaptiveFastParallelMerge<K, As...>(intervals, resultsMutex, dop,
                                                               resultsVec, aggregators...);
  } else {

    for(auto& elem : resultsVec) {
      results.push(std::move(elem));
    }

    auto& threadPool = ThreadPool::getInstance(dop);

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
                          mergeInput2 = std::move(mergeInput2), &results,
                          aggregators...]() mutable {
        groupByAdaptiveParallelPerformMerge<K, As...>(resultsMutex, cv, std::move(mergeInput1),
                                                      std::move(mergeInput2), results,
                                                      aggregators...);
      });
    }

    std::unique_lock<std::mutex> lock(resultsMutex);
    cv.wait(lock, [&results] { return results.size() == 1; });

    resultVectors = std::move(results.front());
    results.pop();
  }

  std::vector<ExpressionSpanArguments> result;
  result.reserve(sizeof...(As) + 1 + secondKey);

  auto [keys1, keys2, payloads] = std::move(resultVectors);
  auto keysPtr1 = std::make_shared<std::vector<K>>(std::move(keys1));
  auto keysPtr2 = std::make_shared<std::vector<K>>(std::move(keys2));
  auto payloadsPtrs = std::apply(
      [&](auto&&... vecs) {
        return std::make_tuple(std::make_shared<std::vector<As>>(std::move(vecs))...);
      },
      std::move(payloads));

  size_t numResultSpans = (keysPtr1->size() + minPartitionSize - 1) / minPartitionSize;

  if(!secondKey) {
    ExpressionSpanArguments keyResultSpans;
    keyResultSpans.reserve(numResultSpans);
    size_t spanStart = 0;
    size_t spanSize = minPartitionSize;
    for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
      keyResultSpans.emplace_back(
          Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
      spanStart += spanSize;
    }
    spanSize = keysPtr1->size() - spanStart;
    if(spanSize > 0) {
      keyResultSpans.emplace_back(
          Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
    }
    result.push_back(std::move(keyResultSpans));
  } else {
    keysPtr2->reserve(keysPtr1->size());
    for(size_t i = 0; i < keysPtr1->size(); ++i) {
      keysPtr2->push_back((*keysPtr1)[i] >> FIRST_KEY_BITS);
      (*keysPtr1)[i] = (*keysPtr1)[i] & FIRST_KEY_MASK;
    }
    ExpressionSpanArguments keyResultSpans1;
    ExpressionSpanArguments keyResultSpans2;
    keyResultSpans1.reserve(numResultSpans);
    keyResultSpans2.reserve(numResultSpans);
    size_t spanStart = 0;
    size_t spanSize = minPartitionSize;
    for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
      keyResultSpans1.emplace_back(
          Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
      keyResultSpans2.emplace_back(
          Span<K>(keysPtr2->data() + spanStart, spanSize, [ptr = keysPtr2]() {}));
      spanStart += spanSize;
    }
    spanSize = keysPtr2->size() - spanStart;
    if(spanSize > 0) {
      keyResultSpans1.emplace_back(
          Span<K>(keysPtr1->data() + spanStart, spanSize, [ptr = keysPtr1]() {}));
      keyResultSpans2.emplace_back(
          Span<K>(keysPtr2->data() + spanStart, spanSize, [ptr = keysPtr2]() {}));
    }
    result.push_back(std::move(keyResultSpans1));
    result.push_back(std::move(keyResultSpans2));
  }

  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (
        [&]() {
          ExpressionSpanArguments payloadSpans;
          payloadSpans.reserve(numResultSpans);
          auto payloadPtr = std::get<Is>(payloadsPtrs);
          size_t spanStart = 0;
          size_t spanSize = minPartitionSize;
          for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
            payloadSpans.emplace_back(
                Span<As>(payloadPtr->data() + spanStart, spanSize, [ptr = payloadPtr]() {}));
            spanStart += spanSize;
          }
          spanSize = payloadPtr->size() - spanStart;
          if(spanSize > 0) {
            payloadSpans.emplace_back(
                Span<As>(payloadPtr->data() + spanStart, spanSize, [ptr = payloadPtr]() {}));
          }
          result.push_back(std::move(payloadSpans));
        }(),
        ...);
  }
  (std::make_index_sequence<sizeof...(As)>());

  return result;
}

template <typename K, typename... As>
std::vector<ExpressionSpanArguments>
groupByAdaptiveParallel(int dop, int cardinality, bool secondKey,
                        ExpressionSpanArguments&& keySpans1, ExpressionSpanArguments&& keySpans2,
                        std::vector<Span<As>>&&... typedAggCols, Aggregator<As>... aggregators) {

  auto& threadPool = ThreadPool::getInstance(dop);
  auto& synchroniser = Synchroniser::getInstance();
  assert(threadPool.getNumThreads() >= dop); // Will have a deadlock otherwise

  int n = 0;
  std::vector<int> spanSizes;
  spanSizes.reserve(keySpans1.size());
  for(const auto& untypedSpan : keySpans1) {
    auto& keySpan1 = std::get<Span<K>>(untypedSpan);
    n += keySpan1.size();
    spanSizes.push_back(keySpan1.size());
  }

  dop = convertToValidDopValue(std::min(dop, n));
  std::mutex resultsMutex;
  std::queue<AggregatedKeysAndPayload<K, As...>> results;
  std::condition_variable cv;

  int tuplesPerThreadBaseline = n / dop;
  int remainingTuples = n % dop;
  int outerIndex = 0;
  int innerIndex = 0;
  int tuplesPerThread; // NOLINT

  for(auto taskNum = 0; taskNum < dop; ++taskNum) {
    tuplesPerThread = tuplesPerThreadBaseline + (taskNum < remainingTuples);

    threadPool.enqueue([&synchroniser, dop, &results, &resultsMutex, &spanSizes, tuplesPerThread,
                        outerIndex, innerIndex, cardinality, secondKey, &keySpans1, &keySpans2,
                        &typedAggCols..., aggregators...] {
      K largestKey = std::numeric_limits<K>::lowest();
      auto aggregatedKeysAndPayload = groupByAdaptive<K, As...>(
          dop, spanSizes, tuplesPerThread, outerIndex, innerIndex, cardinality, secondKey, false,
          largestKey, keySpans1, keySpans2, typedAggCols..., aggregators...);

      if(largestKey > 0) { // We only used hash-based Group, so need to sort result
        std::tuple<As*...> payloadPtrs = [&]<size_t... Is>(std::index_sequence<Is...>) {
          return std::make_tuple(std::get<Is>(std::get<2>(aggregatedKeysAndPayload)).data()...);
        }
        (std::make_index_sequence<sizeof...(As)>());
        sortByKey<K, As...>(std::get<0>(aggregatedKeysAndPayload).size(), largestKey,
                            std::get<0>(aggregatedKeysAndPayload).data(), payloadPtrs);
      }

      {
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.push(std::move(aggregatedKeysAndPayload));
      }

      synchroniser.taskComplete();
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
  synchroniser.waitUntilComplete(dop);

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
    return groupBySort<K, As...>(1, cardinality, numKeys == 2, std::move(keySpans1),
                                 std::move(keySpans2), std::move(typedAggCols)..., aggregators...);
  case Group::GroupAdaptive:
    assert(dop == 1);
    return groupByAdaptive<K, As...>(cardinality, numKeys == 2, std::move(keySpans1),
                                     std::move(keySpans2), std::move(typedAggCols)...,
                                     aggregators...);
  case Group::GroupAdaptiveParallel:
    return groupByAdaptiveParallel<K, As...>(dop, cardinality, numKeys == 2, std::move(keySpans1),
                                             std::move(keySpans2), std::move(typedAggCols)...,
                                             aggregators...);
  default:
    throw std::runtime_error("Invalid selection of 'Group' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_GROUPIMPLEMENTATION_HPP

// NOLINTEND(readability-function-cognitive-complexity)