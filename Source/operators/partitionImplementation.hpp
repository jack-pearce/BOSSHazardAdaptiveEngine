#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "constants/machineConstants.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/systemInformation.hpp"

#define ADAPTIVITY_OUTPUT
#define CHANGE_PARTITION_TO_SORT

namespace adaptive {

/****************************** FORWARD DECLARATIONS ******************************/

template <typename T> class MonitorPartition {
public:
  explicit MonitorPartition(const long_long* sTlbStoreMisses_)
      : sTlbStoreMisses(sTlbStoreMisses_), tuplesPerTlbStoreMiss(50.0) {}
  inline bool robustnessIncreaseRequired(int tuplesProcessed) {
    return (static_cast<float>(tuplesProcessed) / *sTlbStoreMisses) < tuplesPerTlbStoreMiss;
  }

private:
  const long_long* sTlbStoreMisses;
  float tuplesPerTlbStoreMiss;
};

/****************************** FOUNDATIONAL ALGORITHMS ******************************/

template <typename T> struct PartitionedArray {
  std::shared_ptr<T[]> partitionedKeys;
  std::shared_ptr<int32_t[]> indexes;
  std::unique_ptr<std::vector<int>> partitionPositions;
};

template <typename T> class Partition {
public:
  Partition(int n, T* keys, int radixBitsInput = 0) : nInput(n), keysInput(keys) {
    static_assert(std::is_integral<T>::value, "PartitionOperators column must be an integer type");

    std::string minName = "Partition_minRadixBits";
    auto radixBitsMin =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));
    radixBitsOperator = std::max(radixBitsInput, radixBitsMin);

    T largest = std::numeric_limits<T>::min();
    for(int i = 0; i < n; i++) {
      largest = std::max(largest, keys[i]);
    }

    msbToPartitionInput = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartitionInput++;
    }

#ifdef CHANGE_PARTITION_TO_SORT
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l3cacheSize()) / (sizeof(T) * 2 * 2.5);
#endif

    buckets = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer = std::make_shared<T[]>(n);
    tmpBuffer = std::make_unique<T[]>(n);

    returnIndexes = std::make_shared<int32_t[]>(n);
    tmpIndexes = std::make_unique<int32_t[]>(n);

    auto indexesPtr = tmpIndexes.get();
    for(auto i = 0; i < n; ++i) {
      indexesPtr[i] = i;
    }
  }

  PartitionedArray<T> processInput() {
    performPartition(nInput, keysInput, returnBuffer.get(), tmpIndexes.get(), returnIndexes.get(),
                     msbToPartitionInput, radixBitsOperator, 0, false, true);
    return PartitionedArray<T>{returnBuffer, returnIndexes,
                               std::make_unique<std::vector<int>>(std::move(outputPartitions))};
  }

  inline void performPartition(int n, T* keys, T* buffer, int32_t* indexes, int32_t* indexesBuffer,
                               int msbToPartition, int radixBits, int offset, bool copyRequired,
                               bool firstPass = false) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    int i;
    for(i = 0; i < n; i++) {
      buckets[1 + ((keys[i] >> shifts) & mask)]++;
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets[i] += buckets[i - 1];
    }

    std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);

    for(i = 0; i < n; i++) {
      auto index = buckets[(keys[i] >> shifts) & mask]++;
      buffer[index] = keys[i];
      indexesBuffer[index] = indexes[i];
    }

    std::fill(buckets.begin(), buckets.end(), 0);
    msbToPartition -= radixBits;

    if(firstPass) {
      keys = tmpBuffer.get(); // Swap to using temporary buffer to leave original array unmodified
    }

    if(msbToPartition == 0) { // No ability to partition further, so return early
      if(copyRequired) {
        std::memcpy(keys, buffer, n * sizeof(T));
        std::memcpy(indexes, indexesBuffer, n * sizeof(int32_t));
      }
      int prevPartitionEnd = 0;
      for(int partitionEnd : partitions) {
        if(partitionEnd != prevPartitionEnd) {
          outputPartitions.push_back(offset + partitionEnd);
        }
        prevPartitionEnd = partitionEnd;
      }
      return;
    }

    int prevPartitionEnd = 0;
    for(int partitionEnd : partitions) {
      if((partitionEnd - prevPartitionEnd) > maxElementsPerPartition) {
        performPartition(partitionEnd - prevPartitionEnd, buffer + prevPartitionEnd,
                         keys + prevPartitionEnd, indexesBuffer + prevPartitionEnd,
                         indexes + prevPartitionEnd, msbToPartition, radixBits,
                         offset + prevPartitionEnd, !copyRequired);
      } else {
        if(copyRequired) {
          std::memcpy(keys + prevPartitionEnd, buffer + prevPartitionEnd,
                      (partitionEnd - prevPartitionEnd) * sizeof(T));
          std::memcpy(indexes + prevPartitionEnd, indexesBuffer + prevPartitionEnd,
                      (partitionEnd - prevPartitionEnd) * sizeof(int32_t));
        }
        if(partitionEnd != prevPartitionEnd) {
          outputPartitions.push_back(offset + partitionEnd);
        }
      }
      prevPartitionEnd = partitionEnd;
    }
  }

private:
  int nInput;
  T* keysInput;
  std::shared_ptr<T[]> returnBuffer;
  std::unique_ptr<T[]> tmpBuffer;
  std::shared_ptr<int32_t[]> returnIndexes;
  std::unique_ptr<int32_t[]> tmpIndexes;
  int radixBitsOperator;
  int msbToPartitionInput;

  int maxElementsPerPartition;
  std::vector<int> buckets;
  std::vector<int> outputPartitions;
};

/****************************** SINGLE-THREADED ******************************/

static inline PAPI_eventSet& getDataTlbStoreMissesEventSet() {
  thread_local static PAPI_eventSet eventSet({"DTLB-STORE-MISSES"});
  return eventSet;
}

template <typename T> class PartitionAdaptive {
public:
  PartitionAdaptive(int n, T* keys)
      : nInput(n), keysInput(keys), eventSet(getDataTlbStoreMissesEventSet()),
        monitor(MonitorPartition<T>(eventSet.getCounterDiffsPtr())) {
    static_assert(std::is_integral<T>::value, "PartitionOperators column must be an integer type");

    std::string startName = "Partition_startRadixBits";
    radixBitsOperator =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(startName));

    std::string minName = "Partition_minRadixBits";
    minimumRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));

    T largest = std::numeric_limits<T>::min();
    for(int i = 0; i < n; i++) {
      largest = std::max(largest, keys[i]);
    }

    msbToPartitionInput = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartitionInput++;
    }

#ifdef CHANGE_PARTITION_TO_SORT
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l3cacheSize()) / (sizeof(T) * 2 * 2.5);
#endif

    buckets = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer = std::make_shared<T[]>(n);
    tmpBuffer = std::make_unique<T[]>(n);

    returnIndexes = std::make_shared<int32_t[]>(n);
    tmpIndexes = std::make_unique<int32_t[]>(n);

    auto indexesPtr = tmpIndexes.get();
    for(auto i = 0; i < n; ++i) {
      indexesPtr[i] = i;
    }
  }

  PartitionedArray<T> processInput() {
    tuplesPerHazardCheck = 10 * 1000;
    performPartition(nInput, keysInput, returnBuffer.get(), tmpIndexes.get(), returnIndexes.get(),
                     msbToPartitionInput, radixBitsOperator, 0, false, true);
    return PartitionedArray<T>{returnBuffer, returnIndexes,
                               std::make_unique<std::vector<int>>(std::move(outputPartitions))};
  }

  inline void performPartition(int n, T* keys, T* buffer, int32_t* indexes, int32_t* indexesBuffer,
                               int msbToPartition, int radixBits, int offset, bool copyRequired,
                               bool firstPass = false) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    int i, microBatchStart, microBatchSize;
    for(i = 0; i < n; i++) {
      buckets[1 + ((keys[i] >> shifts) & mask)]++;
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets[i] += buckets[i - 1];
    }

    std::vector<int> partitions(buckets.data() + 1, buckets.data() + numBuckets + 1);

    i = 0;
    if(radixBits > minimumRadixBits) {
      while(i < n) {
        microBatchSize = std::min(tuplesPerHazardCheck, n - i);
        microBatchStart = i;

        eventSet.readCounters();

        for(; i < microBatchStart + microBatchSize; i++) { // Run chunk
          auto index = buckets[(keys[i] >> shifts) & mask]++;
          buffer[index] = keys[i];
          indexesBuffer[index] = indexes[i];
        }

        eventSet.readCountersAndUpdateDiff();

        if(monitor.robustnessIncreaseRequired(microBatchSize)) {
          --radixBits;
          ++shifts;
          numBuckets >>= 1;
          mask = numBuckets - 1;

#ifdef ADAPTIVITY_OUTPUT
          std::cout << "RadixBits reduced to " << radixBitsOperator << " after tuple " << i
                    << " due to reading of ";
          std::cout << (static_cast<float>(microBatchSize) /
                        static_cast<float>(*(eventSet.getCounterDiffsPtr())));
          std::cout << " tuples per TLB store miss" << std::endl;
#endif

          mergePartitions(buffer, indexesBuffer, partitions, numBuckets);

          if(radixBits == minimumRadixBits) { // Complete partitioning to avoid unnecessary checks
            for(; i < n; i++) {
              auto index = buckets[(keys[i] >> shifts) & mask]++;
              buffer[index] = keys[i];
              indexesBuffer[index] = indexes[i];
            }
            break;
          }
        }
      }
    } else {
      for(; i < n; i++) {
        auto index = buckets[(keys[i] >> shifts) & mask]++;
        buffer[index] = keys[i];
        indexesBuffer[index] = indexes[i];
      }
    }

    std::fill(buckets.begin(), buckets.end(), 0);
    msbToPartition -= radixBits;

    if(firstPass) {
      keys = tmpBuffer.get(); // Swap to using temporary buffer to leave original array unmodified
    }

    if(msbToPartition == 0) { // No ability to partition further, so return early
      if(copyRequired) {
        std::memcpy(keys, buffer, n * sizeof(T));
        std::memcpy(indexes, indexesBuffer, n * sizeof(int32_t));
      }
      int prevPartitionEnd = 0;
      for(int partitionEnd : partitions) {
        if(partitionEnd != prevPartitionEnd) {
          outputPartitions.push_back(offset + partitionEnd);
        }
        prevPartitionEnd = partitionEnd;
      }
      return;
    }

    int prevPartitionEnd = 0;
    for(int partitionEnd : partitions) {
      if((partitionEnd - prevPartitionEnd) > maxElementsPerPartition) {
        performPartition(partitionEnd - prevPartitionEnd, buffer + prevPartitionEnd,
                         keys + prevPartitionEnd, indexesBuffer + prevPartitionEnd,
                         indexes + prevPartitionEnd, msbToPartition, radixBits,
                         offset + prevPartitionEnd, !copyRequired);
      } else {
        if(copyRequired) {
          std::memcpy(keys + prevPartitionEnd, buffer + prevPartitionEnd,
                      (partitionEnd - prevPartitionEnd) * sizeof(T));
          std::memcpy(indexes + prevPartitionEnd, indexesBuffer + prevPartitionEnd,
                      (partitionEnd - prevPartitionEnd) * sizeof(int32_t));
        }
        if(partitionEnd != prevPartitionEnd) {
          outputPartitions.push_back(offset + partitionEnd);
        }
      }
      prevPartitionEnd = partitionEnd;
    }
  }

  inline void mergePartitions(T* buffer, int32_t* indexesBuffer, std::vector<int>& partitions, int numBuckets) {
    for(int j = 0; j < numBuckets; ++j) { // Move values in buffer
      auto destIndex = buckets[j << 1];
      auto srcIndex = partitions[j << 1];
      auto numElements = buckets[(j << 1) + 1] - srcIndex;
      std::memcpy(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(T));
      std::memcpy(&indexesBuffer[destIndex], &indexesBuffer[srcIndex], numElements * sizeof(T));
    }

    for(int j = 0; j < numBuckets; ++j) { // Merge histogram values and reduce size
      buckets[j] = buckets[j << 1] + (buckets[(j << 1) + 1] - partitions[j << 1]);
    }
    buckets.resize(1 + numBuckets);

    for(int j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j - 1] = partitions[(j << 1) - 1];
    }
    partitions.resize(numBuckets);
  }

private:
  int nInput;
  T* keysInput;
  std::shared_ptr<T[]> returnBuffer;
  std::unique_ptr<T[]> tmpBuffer;
  std::shared_ptr<int32_t[]> returnIndexes;
  std::unique_ptr<int32_t[]> tmpIndexes;
  int minimumRadixBits;
  int radixBitsOperator;
  int msbToPartitionInput;

  int maxElementsPerPartition;
  std::vector<int> buckets;
  std::vector<int> outputPartitions;

  PAPI_eventSet& eventSet;
  MonitorPartition<T> monitor;
  int tuplesPerHazardCheck{};
};

/****************************** ENTRY FUNCTION ******************************/

template <typename T>
std::vector<int> partition(PartitionOperators partitionImplementation, int n, T* keys,
                           int radixBits) {
  if(partitionImplementation == PartitionOperators::RadixBitsFixed) {
    Partition<T> partitionOperator(n, keys, radixBits);
    return partitionOperator.processInput();
  }
  if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
    PartitionAdaptive<T> partitionOperator(n, keys);
    return partitionOperator.processInput();
  }
  throw std::runtime_error("Invalid selection of 'Partition' implementation!");
}

/*********************************** JOIN ***********************************/

// TODO - to update to ensure 1-1 mapping. Need to start with msb
template <typename T1, typename T2> class PartitionJoinExprOperator {
public:
  PartitionJoinExprOperator(Span<T1>& tableOneKeys, Span<T2>& tableTwoKeys)
      : keysOneOperator(Partition<T1>(tableOneKeys.size(), &(*tableOneKeys.begin()))),
        keysTwoOperator(Partition<T2>(tableTwoKeys.size(), &(*tableTwoKeys.begin()))) {}

  PartitionedJoinArguments processInput() {
    auto partitionedTableOne = keysOneOperator.processInput();
    auto keysOne = partitionedTableOne.partitionedKeys.get();
    auto indexesOne = partitionedTableOne.indexes.get();

    auto partitionedTableTwo = keysTwoOperator.processInput();
    auto keysTwo = partitionedTableTwo.partitionedKeys.get();
    auto indexesTwo = partitionedTableTwo.indexes.get();

    ExpressionSpanArguments tableOneKeySpans, tableOneIndexSpans, tableTwoKeySpans,
        tableTwoIndexSpans;

    int prevPartitionEnd = 0;
    for(int partitionEnd : *(partitionedTableOne.partitionPositions.get())) {
      tableOneKeySpans.emplace_back(Span<T1>(keysOne + prevPartitionEnd,
                                             partitionEnd - prevPartitionEnd,
                                             [ptr = partitionedTableOne.partitionedKeys]() {}));
      tableOneIndexSpans.emplace_back(Span<int32_t>(indexesOne + prevPartitionEnd,
                                                    partitionEnd - prevPartitionEnd,
                                                    [ptr = partitionedTableOne.indexes]() {}));
      prevPartitionEnd = partitionEnd;
    }

    prevPartitionEnd = 0;
    for(int partitionEnd : *(partitionedTableTwo.partitionPositions.get())) {
      tableTwoKeySpans.emplace_back(Span<T2>(keysTwo + prevPartitionEnd,
                                             partitionEnd - prevPartitionEnd,
                                             [ptr = partitionedTableTwo.partitionedKeys]() {}));
      tableTwoIndexSpans.emplace_back(Span<int32_t>(indexesTwo + prevPartitionEnd,
                                                    partitionEnd - prevPartitionEnd,
                                                    [ptr = partitionedTableTwo.indexes]() {}));
      prevPartitionEnd = partitionEnd;
    }

    return {std::move(tableOneKeySpans), std::move(tableOneIndexSpans), std::move(tableTwoKeySpans),
            std::move(tableTwoIndexSpans)};
  }

private:
  Partition<T1> keysOneOperator;
  Partition<T2> keysTwoOperator;
};

// TODO - to update so it ladders. Need to start with msb
template <typename T1, typename T2> class PartitionJoinExprOperatorAdaptive {
public:
  PartitionJoinExprOperatorAdaptive(Span<T1>& tableOneKeys, Span<T2>& tableTwoKeys)
      : keysOneOperator(PartitionAdaptive<T1>(tableOneKeys.size(), &(*tableOneKeys.begin()))),
        keysTwoOperator(PartitionAdaptive<T2>(tableTwoKeys.size(), &(*tableTwoKeys.begin()))) {}

  PartitionedJoinArguments processInput() {
    auto partitionedTableOne = keysOneOperator.processInput();
    auto keysOne = partitionedTableOne.partitionedKeys.get();
    auto indexesOne = partitionedTableOne.indexes.get();

    auto partitionedTableTwo = keysTwoOperator.processInput();
    auto keysTwo = partitionedTableTwo.partitionedKeys.get();
    auto indexesTwo = partitionedTableTwo.indexes.get();

    ExpressionSpanArguments tableOneKeySpans, tableOneIndexSpans, tableTwoKeySpans,
        tableTwoIndexSpans;

    int prevPartitionEnd = 0;
    for(int partitionEnd : *(partitionedTableOne.partitionPositions.get())) {
      tableOneKeySpans.emplace_back(Span<T1>(keysOne + prevPartitionEnd,
                                             partitionEnd - prevPartitionEnd,
                                             [ptr = partitionedTableOne.partitionedKeys]() {}));
      tableOneIndexSpans.emplace_back(Span<int32_t>(indexesOne + prevPartitionEnd,
                                                    partitionEnd - prevPartitionEnd,
                                                    [ptr = partitionedTableOne.indexes]() {}));
      prevPartitionEnd = partitionEnd;
    }

    prevPartitionEnd = 0;
    for(int partitionEnd : *(partitionedTableTwo.partitionPositions.get())) {
      tableTwoKeySpans.emplace_back(Span<T2>(keysTwo + prevPartitionEnd,
                                             partitionEnd - prevPartitionEnd,
                                             [ptr = partitionedTableTwo.partitionedKeys]() {}));
      tableTwoIndexSpans.emplace_back(Span<int32_t>(indexesTwo + prevPartitionEnd,
                                                    partitionEnd - prevPartitionEnd,
                                                    [ptr = partitionedTableTwo.indexes]() {}));
      prevPartitionEnd = partitionEnd;
    }

    return {std::move(tableOneKeySpans), std::move(tableOneIndexSpans), std::move(tableTwoKeySpans),
            std::move(tableTwoIndexSpans)};
  }

private:
  PartitionAdaptive<T1> keysOneOperator;
  PartitionAdaptive<T2> keysTwoOperator;
};

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           Span<T1>& tableOneKeys, Span<T2>& tableTwoKeys) {
  if(partitionImplementation == PartitionOperators::RadixBitsFixed) {
    PartitionJoinExprOperator<T1, T2> partitionJoinExprOperator(tableOneKeys, tableTwoKeys);
    return partitionJoinExprOperator.processInput();
  }
  if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
    PartitionJoinExprOperatorAdaptive<T1, T2> partitionJoinExprOperator(tableOneKeys, tableTwoKeys);
    return partitionJoinExprOperator.processInput();
  }
  throw std::runtime_error("Invalid selection of 'Partition' implementation!");
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
