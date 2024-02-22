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
#include "utilities/dataStructures.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/systemInformation.hpp"

#define ADAPTIVITY_OUTPUT
#define CHANGE_PARTITION_TO_SORT

namespace adaptive {

/****************************** FORWARD DECLARATIONS ******************************/

class MonitorPartition {
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
  std::unique_ptr<vectorOfPairs<int, int>> partitionPositions; // {start, size}
};

template <typename T1, typename T2> struct TwoPartitionedArrays {
  PartitionedArray<T1> partitionedArrayOne;
  PartitionedArray<T2> partitionedArrayTwo;
};

template <typename T1, typename T2> class Partition {
public:
  Partition(int n1, T1* keys1, int n2, T2* keys2, int radixBitsInput = 0)
      : nInput1(n1), keysInput1(keys1), nInput2(n2), keysInput2(keys2) {
    std::string minName = "Partition_minRadixBits";
    auto radixBitsMin =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));
    radixBitsOperator = std::max(radixBitsInput, radixBitsMin);

    msbToPartitionInput = std::max(getMsb(n1, keys1), getMsb(n2, keys2));

#ifdef CHANGE_PARTITION_TO_SORT
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l3cacheSize()) / (sizeof(T) * 2 * 2.5);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared<T1[]>(n1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared<int32_t[]>(n1);
    tmpIndexes1 = std::make_unique<int32_t[]>(n1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < n1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(2 + (2 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared<T2[]>(n2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared<int32_t[]>(n2);
    tmpIndexes2 = std::make_unique<int32_t[]>(n2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < n2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<T1, T2> processInput() {
    performPartition(nInput1, keysInput1, returnBuffer1.get(), tmpIndexes1.get(),
                     returnIndexes1.get(), 0, nInput2, keysInput2, returnBuffer2.get(),
                     tmpIndexes2.get(), returnIndexes2.get(), 0, msbToPartitionInput,
                     radixBitsOperator, false, true);
    return TwoPartitionedArrays<T1, T2>{
        PartitionedArray<T1>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<T2>{
            returnBuffer2, returnIndexes2,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))}};
  }

private:
  inline void performPartition(int n1, T1* keys1, T1* buffer1, int32_t* indexes1,
                               int32_t* indexesBuffer1, int offset1, int n2, T2* keys2, T2* buffer2,
                               int32_t* indexes2, int32_t* indexesBuffer2, int offset2,
                               int msbToPartition, int radixBits, bool copyRequired,
                               bool firstPass = false) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    // Complete partitioning for array 1
    int i;
    for(i = 0; i < n1; i++) {
      buckets1[1 + ((keys1[i] >> shifts) & mask)]++;
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);
    for(i = 0; i < n1; i++) {
      auto index = buckets1[(keys1[i] >> shifts) & mask]++;
      buffer1[index] = keys1[i];
      indexesBuffer1[index] = indexes1[i];
    }
    std::fill(buckets1.begin(), buckets1.end(), 0);

    // Complete partitioning for array 2
    for(i = 0; i < n2; i++) {
      buckets2[1 + ((keys2[i] >> shifts) & mask)]++;
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);
    for(i = 0; i < n2; i++) {
      auto index = buckets2[(keys2[i] >> shifts) & mask]++;
      buffer2[index] = keys2[i];
      indexesBuffer2[index] = indexes2[i];
    }
    std::fill(buckets2.begin(), buckets2.end(), 0);

    msbToPartition -= radixBits;

    if(msbToPartition == 0) { // No ability to partition further, so return early
      if(copyRequired) {      // Will not be called on first pass
        std::memcpy(keys1, buffer1, n1 * sizeof(T1));
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int32_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int32_t));
      }

      int prevPartitionEnd1 = 0;
      int prevPartitionEnd2 = 0;
      for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(offset1 + prevPartitionEnd1,
                                         partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(offset2 + prevPartitionEnd2,
                                         partitions2[j] - prevPartitionEnd2);
        }
        prevPartitionEnd1 = partitions1[j];
        prevPartitionEnd2 = partitions2[j];
      }
      return;
    }

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          if(firstPass && tmpBuffer1 == nullptr) {
            tmpBuffer1 = std::make_unique<T1[]>(n1); // Lazily allocate tmpBuffer
            tmpBuffer2 = std::make_unique<T2[]>(n2);
            keys1 = tmpBuffer1.get(); // Use tmp buffer to leave original array unmodified
            keys2 = tmpBuffer2.get();
          }
          performPartition(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                           keys1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                           indexes1 + prevPartitionEnd1, offset1 + prevPartitionEnd1,
                           partitions2[j] - prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                           keys2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                           indexes2 + prevPartitionEnd2, offset2 + prevPartitionEnd2,
                           msbToPartition, radixBits, !copyRequired);
        } else {
          if(copyRequired) { // Will not be called on first pass
            std::memcpy(keys1 + prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                        (partitions1[j] - prevPartitionEnd1) * sizeof(T1));
            std::memcpy(indexes1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                        (partitions1[j] - prevPartitionEnd1) * sizeof(int32_t));
            std::memcpy(keys2 + prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(T2));
            std::memcpy(indexes2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(int32_t));
          }
          outputPartitions1.emplace_back(offset1 + prevPartitionEnd1,
                                         partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(offset2 + prevPartitionEnd2,
                                         partitions2[j] - prevPartitionEnd2);
        }
      }
      prevPartitionEnd1 = partitions1[j];
      prevPartitionEnd2 = partitions2[j];
    }
  }

  template <typename U> inline int getMsb(int n, U* keys) {
    U largest = std::numeric_limits<U>::min();
    for(int i = 0; i < n; i++) {
      largest = std::max(largest, keys[i]);
    }

    msbToPartitionInput = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartitionInput++;
    }
    return msbToPartitionInput;
  }

  int nInput1;
  T1* keysInput1;
  std::shared_ptr<T1[]> returnBuffer1;
  std::unique_ptr<T1[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  T2* keysInput2;
  std::shared_ptr<T2[]> returnBuffer2;
  std::unique_ptr<T2[]> tmpBuffer2;
  std::shared_ptr<int32_t[]> returnIndexes2;
  std::unique_ptr<int32_t[]> tmpIndexes2;
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;

  int radixBitsOperator;
  int msbToPartitionInput;
  int maxElementsPerPartition;
};

/****************************** SINGLE-THREADED ******************************/

static inline PAPI_eventSet& getDataTlbStoreMissesEventSet() {
  thread_local static PAPI_eventSet eventSet({"DTLB-STORE-MISSES"});
  return eventSet;
}

template <typename T1, typename T2> class PartitionAdaptive {
public:
  PartitionAdaptive(int n1, T1* keys1, int n2, T2* keys2)
      : nInput1(n1), keysInput1(keys1), nInput2(n2), keysInput2(keys2),
        eventSet(getDataTlbStoreMissesEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr())) {
    std::string startName = "Partition_startRadixBits";
    radixBitsOperator =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(startName));

    std::string minName = "Partition_minRadixBits";
    minimumRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));

    msbToPartitionInput = std::max(getMsb(n1, keys1), getMsb(n2, keys2));

#ifdef CHANGE_PARTITION_TO_SORT
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l3cacheSize()) / (sizeof(T) * 2 * 2.5);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared<T1[]>(n1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared<int32_t[]>(n1);
    tmpIndexes1 = std::make_unique<int32_t[]>(n1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < n1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(2 + (2 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared<T2[]>(n2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared<int32_t[]>(n2);
    tmpIndexes2 = std::make_unique<int32_t[]>(n2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < n2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<T1, T2> processInput() {
    tuplesPerHazardCheck = 10 * 1000;
    performPartition(nInput1, keysInput1, returnBuffer1.get(), tmpIndexes1.get(),
                     returnIndexes1.get(), 0, nInput2, keysInput2, returnBuffer2.get(),
                     tmpIndexes2.get(), returnIndexes2.get(), 0, msbToPartitionInput,
                     radixBitsOperator, false, true);
    return TwoPartitionedArrays<T1, T2>{
        PartitionedArray<T1>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<T2>{
            returnBuffer2, returnIndexes2,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))}};
  }

private:
  inline void performPartition(int n1, T1* keys1, T1* buffer1, int32_t* indexes1,
                               int32_t* indexesBuffer1, int offset1, int n2, T2* keys2, T2* buffer2,
                               int32_t* indexes2, int32_t* indexesBuffer2, int offset2,
                               int msbToPartition, int radixBits, bool copyRequired,
                               bool firstPass = false) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    // Complete histogram for array 1
    int i, microBatchStart, microBatchSize;
    for(i = 0; i < n1; i++) {
      buckets1[1 + ((keys1[i] >> shifts) & mask)]++;
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);

    // Complete histogram for array 2
    for(i = 0; i < n2; i++) {
      buckets2[1 + ((keys2[i] >> shifts) & mask)]++;
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);

    int i1 = 0, i2 = 0;
    if(radixBits > minimumRadixBits) {
      while(i2 < n2 && i1 < n1) { // NOLINT
        if(i1 < n1) {             // NOLINT
          microBatchSize = std::min(tuplesPerHazardCheck, n1 - i1);
          microBatchStart = i1;

          processMicroBatch<T1, T2>(microBatchStart, microBatchSize, i1, n1, keys1, buffer1,
                                    indexes1, indexesBuffer1, buckets1, partitions1, shifts, mask,
                                    radixBits, numBuckets, i2, n2, keys2, buffer2, indexes2,
                                    indexesBuffer2, buckets2, partitions2);
        }
        if(i2 < n2) { // NOLINT
          microBatchSize = std::min(tuplesPerHazardCheck, n2 - i2);
          microBatchStart = i2;

          processMicroBatch<T2, T1>(microBatchStart, microBatchSize, i2, n2, keys2, buffer2,
                                    indexes2, indexesBuffer2, buckets2, partitions2, shifts, mask,
                                    radixBits, numBuckets, i1, n1, keys1, buffer1, indexes1,
                                    indexesBuffer1, buckets1, partitions1);
        }
      }
    } else {
      for(; i1 < n1; i1++) {
        auto index = buckets1[(keys1[i1] >> shifts) & mask]++;
        buffer1[index] = keys1[i1];
        indexesBuffer1[index] = indexes1[i1];
      }
      for(; i2 < n2; i2++) {
        auto index = buckets2[(keys2[i2] >> shifts) & mask]++;
        buffer2[index] = keys2[i2];
        indexesBuffer2[index] = indexes2[i2];
      }
    }

    std::fill(buckets1.begin(), buckets1.end(), 0);
    std::fill(buckets2.begin(), buckets2.end(), 0);
    msbToPartition -= radixBits;

    if(msbToPartition == 0) { // No ability to partition further, so return early
      if(copyRequired) {      // Will not be called on first pass
        std::memcpy(keys1, buffer1, n1 * sizeof(T1));
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int32_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int32_t));
      }

      int prevPartitionEnd1 = 0;
      int prevPartitionEnd2 = 0;
      for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(offset1 + prevPartitionEnd1,
                                         partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(offset2 + prevPartitionEnd2,
                                         partitions2[j] - prevPartitionEnd2);
        }
        prevPartitionEnd1 = partitions1[j];
        prevPartitionEnd2 = partitions2[j];
      }
      return;
    }

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          if(firstPass && tmpBuffer1 == nullptr) {
            tmpBuffer1 = std::make_unique<T1[]>(n1); // Lazily allocate tmpBuffer
            tmpBuffer2 = std::make_unique<T2[]>(n2);
            keys1 = tmpBuffer1.get(); // Use tmp buffer to leave original array unmodified
            keys2 = tmpBuffer2.get();
          }
          performPartition(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                           keys1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                           indexes1 + prevPartitionEnd1, offset1 + prevPartitionEnd1,
                           partitions2[j] - prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                           keys2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                           indexes2 + prevPartitionEnd2, offset2 + prevPartitionEnd2,
                           msbToPartition, radixBits, !copyRequired);
        } else {
          if(copyRequired) { // Will not be called on first pass
            std::memcpy(keys1 + prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                        (partitions1[j] - prevPartitionEnd1) * sizeof(T1));
            std::memcpy(indexes1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                        (partitions1[j] - prevPartitionEnd1) * sizeof(int32_t));
            std::memcpy(keys2 + prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(T2));
            std::memcpy(indexes2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(int32_t));
          }
          outputPartitions1.emplace_back(offset1 + prevPartitionEnd1,
                                         partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(offset2 + prevPartitionEnd2,
                                         partitions2[j] - prevPartitionEnd2);
        }
      }
      prevPartitionEnd1 = partitions1[j];
      prevPartitionEnd2 = partitions2[j];
    }
  }

  template <typename U1, typename U2>
  inline void processMicroBatch(int microBatchStart, int microBatchSize, int& i, int n,
                                const U1* keys, U1* buffer, const int32_t* indexes,
                                int32_t* indexesBuffer, std::vector<int>& buckets,
                                std::vector<int>& partitions, int shifts, unsigned int mask,
                                int& radixBits, int& numBuckets, int& i_2, int n_2,
                                const U2* keys_2, U2* buffer_2, const int32_t* indexes_2,
                                int32_t* indexesBuffer_2, std::vector<int>& buckets_2,
                                std::vector<int>& partitions_2) {
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

      mergePartitions(buffer, indexesBuffer, partitions, buckets, numBuckets, i);
      mergePartitions(buffer_2, indexesBuffer_2, partitions_2, buckets_2, numBuckets, i_2);

      if(radixBits == minimumRadixBits) { // Complete partitioning to avoid unnecessary checks
        for(; i < n; i++) {
          auto index = buckets[(keys[i] >> shifts) & mask]++;
          buffer[index] = keys[i];
          indexesBuffer[index] = indexes[i];
        }
        for(; i_2 < n_2; i_2++) {
          auto index = buckets_2[(keys_2[i_2] >> shifts) & mask]++;
          buffer_2[index] = keys_2[i_2];
          indexesBuffer_2[index] = indexes_2[i_2];
        }
      }
    }
  }

  template <typename U>
  inline void mergePartitions(U* buffer, int32_t* indexesBuffer, std::vector<int>& partitions,
                              std::vector<int>& buckets, int numBuckets, int numProcessed) {
    if(numProcessed > 0) {                  // Skip if no elements have been scattered yet
      for(int j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1];
        auto srcIndex = partitions[j << 1];
        auto numElements = buckets[(j << 1) + 1] - srcIndex;
        std::memcpy(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U));
        std::memcpy(&indexesBuffer[destIndex], &indexesBuffer[srcIndex], numElements * sizeof(U));
      }
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

  template <typename U> inline int getMsb(int n, U* keys) {
    U largest = std::numeric_limits<U>::min();
    for(int i = 0; i < n; i++) {
      largest = std::max(largest, keys[i]);
    }

    msbToPartitionInput = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartitionInput++;
    }
    return msbToPartitionInput;
  }

  int nInput1;
  T1* keysInput1;
  std::shared_ptr<T1[]> returnBuffer1;
  std::unique_ptr<T1[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  T2* keysInput2;
  std::shared_ptr<T2[]> returnBuffer2;
  std::unique_ptr<T2[]> tmpBuffer2;
  std::shared_ptr<int32_t[]> returnIndexes2;
  std::unique_ptr<int32_t[]> tmpIndexes2;
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;

  int minimumRadixBits;
  int radixBitsOperator;
  int msbToPartitionInput;
  int maxElementsPerPartition;

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  int tuplesPerHazardCheck{};
};

/*********************************** ENTRY FUNCTION ***********************************/

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           Span<T1>& tableOneKeys, Span<T2>& tableTwoKeys) {
  static_assert(std::is_integral<T1>::value, "PartitionOperators column must be an integer type");
  static_assert(std::is_integral<T2>::value, "PartitionOperators column must be an integer type");

  TwoPartitionedArrays partitionedTables = [partitionImplementation, &tableOneKeys,
                                            &tableTwoKeys]() {
    if(partitionImplementation == PartitionOperators::RadixBitsFixed) {
      auto partitionOperator = Partition<T1, T2>(tableOneKeys.size(), &(*tableOneKeys.begin()),
                                                 tableTwoKeys.size(), &(*tableTwoKeys.begin()));
      return partitionOperator.processInput();
    } else if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
      auto partitionOperator =
          PartitionAdaptive<T1, T2>(tableOneKeys.size(), &(*tableOneKeys.begin()),
                                    tableTwoKeys.size(), &(*tableTwoKeys.begin()));
      return partitionOperator.processInput();
    } else {
      throw std::runtime_error("Invalid selection of 'Partition' implementation!");
    }
  }();

  auto& partitionedTableOne = partitionedTables.partitionedArrayOne;
  auto& partitionedTableTwo = partitionedTables.partitionedArrayTwo;

  auto keysOne = partitionedTableOne.partitionedKeys.get();
  auto indexesOne = partitionedTableOne.indexes.get();
  auto partitionsOne = *(partitionedTableOne.partitionPositions.get());

  auto keysTwo = partitionedTableTwo.partitionedKeys.get();
  auto indexesTwo = partitionedTableTwo.indexes.get();
  auto partitionsTwo = *(partitionedTableTwo.partitionPositions.get());

  ExpressionSpanArguments tableOneKeySpans, tableOneIndexSpans, tableTwoKeySpans,
      tableTwoIndexSpans;

  for(int i = 0; i < static_cast<int>(partitionsOne.size()); ++i) {
    tableOneKeySpans.emplace_back(Span<T1>(keysOne + partitionsOne[i].first,
                                           partitionsOne[i].second,
                                           [ptr = partitionedTableOne.partitionedKeys]() {}));
    tableOneIndexSpans.emplace_back(Span<int32_t>(indexesOne + partitionsOne[i].first,
                                                  partitionsOne[i].second,
                                                  [ptr = partitionedTableOne.indexes]() {}));
    tableTwoKeySpans.emplace_back(Span<T2>(keysTwo + partitionsTwo[i].first,
                                           partitionsTwo[i].second,
                                           [ptr = partitionedTableTwo.partitionedKeys]() {}));
    tableTwoIndexSpans.emplace_back(Span<int32_t>(indexesTwo + partitionsTwo[i].first,
                                                  partitionsTwo[i].second,
                                                  [ptr = partitionedTableTwo.indexes]() {}));
  }

  return {std::move(tableOneKeySpans), std::move(tableOneIndexSpans), std::move(tableTwoKeySpans),
          std::move(tableTwoIndexSpans)};
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
