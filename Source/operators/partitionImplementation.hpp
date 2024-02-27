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

//#define DEBUG
#define ADAPTIVITY_OUTPUT
//#define CHANGE_PARTITION_TO_SORT_FOR_TESTING

namespace adaptive {

/****************************** FORWARD DECLARATIONS ******************************/

class MonitorPartition {
public:
  explicit MonitorPartition(const long_long* sTlbStoreMisses_)
      : sTlbStoreMisses(sTlbStoreMisses_), tuplesPerTlbStoreMiss(50.0) {}
  inline bool robustnessIncreaseRequired(int tuplesProcessed) {
    return (static_cast<float>(tuplesProcessed) / static_cast<float>(*sTlbStoreMisses)) <
           tuplesPerTlbStoreMiss;
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
  Partition(const ExpressionSpanArguments& keySpans1_, const ExpressionSpanArguments& keySpans2_,
            int radixBitsInput = 0)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_) {
    std::string minName = "Partition_minRadixBits";
    auto radixBitsMin =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));
    radixBitsOperator = std::max(radixBitsInput, radixBitsMin);

    msbToPartitionInput = std::max(getMsb<T1>(keySpans1, nInput1), getMsb<T2>(keySpans2, nInput2));

#ifdef CHANGE_PARTITION_TO_SORT_FOR_TESTING
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / (sizeof(T1) * 2 * 2.5);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared<T1[]>(nInput1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared<int32_t[]>(nInput1);
    tmpIndexes1 = std::make_unique<int32_t[]>(nInput1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < nInput1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared<T2[]>(nInput2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared<int32_t[]>(nInput2);
    tmpIndexes2 = std::make_unique<int32_t[]>(nInput2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < nInput2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<T1, T2> processInput() {
    performPartition();
    return TwoPartitionedArrays<T1, T2>{
        PartitionedArray<T1>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<T2>{
            returnBuffer2, returnIndexes2,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))}};
  }

private:
  inline void performPartition() {
    int msbToPartition = msbToPartitionInput;
    auto buffer1 = returnBuffer1.get();
    auto indexes1 = tmpIndexes1.get();
    auto indexesBuffer1 = returnIndexes1.get();
    auto buffer2 = returnBuffer2.get();
    auto indexes2 = tmpIndexes2.get();
    auto indexesBuffer2 = returnIndexes2.get();

    int radixBits = std::min(msbToPartition, radixBitsOperator);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    // Complete partitioning for array 1
    int i;
    for(auto& span : keySpans1) {
      for(auto& key : std::get<Span<T1>>(span)) {
        buckets1[1 + ((key >> shifts) & mask)]++;
      }
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);

    int offset = 0;
    for(auto& untypedSpan : keySpans1) {
      auto& span = std::get<Span<T1>>(untypedSpan);
      for(i = 0; i < static_cast<int>(span.size()); i++) {
        auto index = buckets1[(span[i] >> shifts) & mask]++;
        buffer1[index] = span[i];
        indexesBuffer1[index] = indexes1[offset + i];
      }
      offset += span.size();
    }
    std::fill(buckets1.begin(), buckets1.end(), 0);

    // Complete partitioning for array 2
    for(auto& span : keySpans2) {
      for(auto& key : std::get<Span<T2>>(span)) {
        buckets2[1 + ((key >> shifts) & mask)]++;
      }
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);

    offset = 0;
    for(auto& untypedSpan : keySpans2) {
      auto& span = std::get<Span<T2>>(untypedSpan);
      for(i = 0; i < static_cast<int>(span.size()); i++) {
        auto index = buckets2[(span[i] >> shifts) & mask]++;
        buffer2[index] = span[i];
        indexesBuffer2[index] = indexes2[offset + i];
      }
      offset += span.size();
    }
    std::fill(buckets2.begin(), buckets2.end(), 0);

    msbToPartition -= radixBits;

    if(msbToPartition == 0) { // No ability to partition further, so return early
      int prevPartitionEnd1 = 0;
      int prevPartitionEnd2 = 0;
      for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
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
          if(tmpBuffer1 == nullptr) {
            tmpBuffer1 = std::make_unique<T1[]>(nInput1); // Lazily allocate tmpBuffer
            tmpBuffer2 = std::make_unique<T2[]>(nInput2);
          }
          performPartitionAux(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                              tmpBuffer1.get() + prevPartitionEnd1,
                              indexesBuffer1 + prevPartitionEnd1, indexes1 + prevPartitionEnd1,
                              prevPartitionEnd1, partitions2[j] - prevPartitionEnd2,
                              buffer2 + prevPartitionEnd2, tmpBuffer2.get() + prevPartitionEnd2,
                              indexesBuffer2 + prevPartitionEnd2, indexes2 + prevPartitionEnd2,
                              prevPartitionEnd2, msbToPartition, radixBits, true);
        } else {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
      }
      prevPartitionEnd1 = partitions1[j];
      prevPartitionEnd2 = partitions2[j];
    }
  }

  inline void performPartitionAux(int n1, T1* keys1, T1* buffer1, int32_t* indexes1,
                                  int32_t* indexesBuffer1, int offset1, int n2, T2* keys2,
                                  T2* buffer2, int32_t* indexes2, int32_t* indexesBuffer2,
                                  int offset2, int msbToPartition, int radixBits,
                                  bool copyRequired) {
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
      if(copyRequired) {
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
          performPartitionAux(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                              keys1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                              indexes1 + prevPartitionEnd1, offset1 + prevPartitionEnd1,
                              partitions2[j] - prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                              keys2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                              indexes2 + prevPartitionEnd2, offset2 + prevPartitionEnd2,
                              msbToPartition, radixBits, !copyRequired);
        } else {
          if(copyRequired) {
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

  template <typename U> inline int getMsb(const ExpressionSpanArguments& keySpans, int& n) {
    auto largest = std::numeric_limits<U>::min();
    for(auto& untypedSpan : keySpans) {
      auto& span = std::get<Span<U>>(untypedSpan);
      n += span.size();
      for(auto& key : span) {
        largest = std::max(largest, key);
      }
    }

    int msbToPartition = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartition++;
    }
    return msbToPartition;
  }

  int nInput1;
  const ExpressionSpanArguments& keySpans1;
  std::shared_ptr<T1[]> returnBuffer1;
  std::unique_ptr<T1[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
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
  PartitionAdaptive(const ExpressionSpanArguments& keySpans1_,
                    const ExpressionSpanArguments& keySpans2_)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_),
        eventSet(getDataTlbStoreMissesEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr())), tuplesPerHazardCheck(10 * 1000) {
    std::string startName = "Partition_startRadixBits";
    radixBitsOperator =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(startName));

    std::string minName = "Partition_minRadixBits";
    minimumRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));

    msbToPartitionInput = std::max(getMsb<T1>(keySpans1, nInput1), getMsb<T2>(keySpans2, nInput2));

#ifdef CHANGE_PARTITION_TO_SORT_FOR_TESTING
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / (sizeof(T1) * 2 * 2.5);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared<T1[]>(nInput1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared<int32_t[]>(nInput1);
    tmpIndexes1 = std::make_unique<int32_t[]>(nInput1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < nInput1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared<T2[]>(nInput2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared<int32_t[]>(nInput2);
    tmpIndexes2 = std::make_unique<int32_t[]>(nInput2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < nInput2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<T1, T2> processInput() {
    performPartition();
    return TwoPartitionedArrays<T1, T2>{
        PartitionedArray<T1>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<T2>{
            returnBuffer2, returnIndexes2,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))}};
  }

private:
  inline void performPartition() {
    int msbToPartition = msbToPartitionInput;
    auto buffer1 = returnBuffer1.get();
    auto indexes1 = tmpIndexes1.get();
    auto indexesBuffer1 = returnIndexes1.get();
    auto buffer2 = returnBuffer2.get();
    auto indexes2 = tmpIndexes2.get();
    auto indexesBuffer2 = returnIndexes2.get();

    int radixBits = std::min(msbToPartition, radixBitsOperator);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    // Complete histogram for array 1
    int i;
    for(auto& span : keySpans1) {
      for(auto& key : std::get<Span<T1>>(span)) {
        buckets1[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);

    // Complete histogram for array 2
    for(auto& span : keySpans2) {
      for(auto& key : std::get<Span<T2>>(span)) {
        buckets2[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);

    bool keysComplete1 = false, keysComplete2 = false;
    auto outerIterator1 = keySpans1.begin();
    auto innerIterator1 = (std::get<Span<T1>>(*outerIterator1)).begin();
    auto outerIterator2 = keySpans2.begin();
    auto innerIterator2 = (std::get<Span<T2>>(*outerIterator2)).begin();
    int offset1 = 0, offset2 = 0;

    while(!keysComplete1 || !keysComplete2) { // NOLINT
      if(!keysComplete1) {                    // NOLINT
        processSpansMicroBatch<T1, T2>(
            outerIterator1, innerIterator1, keySpans1, buffer1, indexes1, indexesBuffer1, buckets1,
            partitions1, offset1, keysComplete1, shifts, mask, radixBits, numBuckets,
            outerIterator2, innerIterator2, keySpans2, buffer2, indexes2, indexesBuffer2, buckets2,
            partitions2, offset2, keysComplete2);
      }
      if(!keysComplete2) { // NOLINT
        processSpansMicroBatch<T2, T1>(
            outerIterator2, innerIterator2, keySpans2, buffer2, indexes2, indexesBuffer2, buckets2,
            partitions2, offset2, keysComplete2, shifts, mask, radixBits, numBuckets,
            outerIterator1, innerIterator1, keySpans1, buffer1, indexes1, indexesBuffer1, buckets1,
            partitions1, offset1, keysComplete1);
      }
    }

    std::fill(buckets1.begin(), buckets1.begin() + numBuckets + 1, 0);
    std::fill(buckets2.begin(), buckets2.begin() + numBuckets + 1, 0);

    msbToPartition -= radixBits;

    if(msbToPartition == 0) { // No ability to partition further, so return early
      int prevPartitionEnd1 = 0;
      int prevPartitionEnd2 = 0;
      for(int j = 0; j < static_cast<int>(partitions1.size()); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
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
          if(tmpBuffer1 == nullptr) {
            tmpBuffer1 = std::make_unique<T1[]>(nInput1); // Lazily allocate tmpBuffer
            tmpBuffer2 = std::make_unique<T2[]>(nInput2);
          }
          performPartitionAux(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                              tmpBuffer1.get() + prevPartitionEnd1,
                              indexesBuffer1 + prevPartitionEnd1, indexes1 + prevPartitionEnd1,
                              prevPartitionEnd1, partitions2[j] - prevPartitionEnd2,
                              buffer2 + prevPartitionEnd2, tmpBuffer2.get() + prevPartitionEnd2,
                              indexesBuffer2 + prevPartitionEnd2, indexes2 + prevPartitionEnd2,
                              prevPartitionEnd2, msbToPartition, radixBits, true);
        } else {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
      }
      prevPartitionEnd1 = partitions1[j];
      prevPartitionEnd2 = partitions2[j];
    }
  }

  inline void performPartitionAux(int n1, T1* keys1, T1* buffer1, int32_t* indexes1,
                                  int32_t* indexesBuffer1, int offset1, int n2, T2* keys2,
                                  T2* buffer2, int32_t* indexes2, int32_t* indexesBuffer2,
                                  int offset2, int msbToPartition, int radixBits, bool copyRequired,
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
      while(i2 < n2 || i1 < n1) { // NOLINT
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

    std::fill(buckets1.begin(), buckets1.begin() + numBuckets + 1, 0);
    std::fill(buckets2.begin(), buckets2.begin() + numBuckets + 1, 0);

    msbToPartition -= radixBits;

    if(msbToPartition == 0) { // No ability to partition further, so return early
      if(copyRequired) {
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
          performPartitionAux(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
                              keys1 + prevPartitionEnd1, indexesBuffer1 + prevPartitionEnd1,
                              indexes1 + prevPartitionEnd1, offset1 + prevPartitionEnd1,
                              partitions2[j] - prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                              keys2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                              indexes2 + prevPartitionEnd2, offset2 + prevPartitionEnd2,
                              msbToPartition, radixBits, !copyRequired);
        } else {
          if(copyRequired) {
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

  template <typename U1, typename U2, typename OuterIt, typename InnerIt, typename OuterIt_2,
            typename InnerIt_2>
  inline void processSpansMicroBatch(
      OuterIt& outerIterator, InnerIt& innerIterator, const ExpressionSpanArguments& keySpans,
      U1* buffer, const int32_t* indexes, int32_t* indexesBuffer, std::vector<int>& buckets,
      std::vector<int>& partitions, int& offset, bool& keysComplete, int& shifts,
      unsigned int& mask, int& radixBits, int& numBuckets, OuterIt_2& outerIterator_2,
      InnerIt_2& innerIterator_2, const ExpressionSpanArguments& keySpans_2, U2* buffer_2,
      const int32_t* indexes_2, int32_t* indexesBuffer_2, std::vector<int>& buckets_2,
      std::vector<int>& partitions_2, int& offset_2, bool& keysComplete_2) {
    int processed = 0;
    int microBatchChunkSize;
    eventSet.readCounters();
    while(processed < tuplesPerHazardCheck) {
      microBatchChunkSize =
          std::min(tuplesPerHazardCheck - processed,
                   static_cast<int>((std::get<Span<U1>>(*outerIterator)).end() - innerIterator));
      for(auto i = 0; i < microBatchChunkSize; ++i) { // Run chunk
        auto index = buckets[((*innerIterator) >> shifts) & mask]++;
        buffer[index] = *(innerIterator++);
        indexesBuffer[index] = indexes[offset + processed++];
      }
      if(innerIterator == (std::get<Span<U1>>(*outerIterator)).end()) {
        if(++outerIterator == keySpans.end()) {
          keysComplete = true;
          break;
        } else {
          innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
        }
      }
    }
    eventSet.readCountersAndUpdateDiff();
    offset += processed;

    if(processed == tuplesPerHazardCheck &&
       monitor.robustnessIncreaseRequired(tuplesPerHazardCheck)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1;
      mask = numBuckets - 1;

#ifdef ADAPTIVITY_OUTPUT
      std::cout << "RadixBits reduced to " << radixBits << " due to reading of ";
      std::cout << (static_cast<float>(tuplesPerHazardCheck) /
                    static_cast<float>(*(eventSet.getCounterDiffsPtr())));
      std::cout << " tuples per TLB store miss" << std::endl;
#endif

      mergePartitions(buffer, indexesBuffer, partitions, buckets, numBuckets, true);
      mergePartitions(buffer_2, indexesBuffer_2, partitions_2, buckets_2, numBuckets, offset_2 > 0);

      if(radixBits == minimumRadixBits) { // Complete partitioning to avoid unnecessary checks
        while(true) {
          while(innerIterator != (std::get<Span<U1>>(*outerIterator)).end()) {
            auto index = buckets[((*innerIterator) >> shifts) & mask]++;
            buffer[index] = *(innerIterator++);
            indexesBuffer[index] = indexes[offset++];
          }
          if(++outerIterator == keySpans.end()) {
            break;
          } else {
            innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
          }
        }

        while(true) {
          while(innerIterator_2 != (std::get<Span<U2>>(*outerIterator_2)).end()) {
            auto index = buckets_2[((*innerIterator_2) >> shifts) & mask]++;
            buffer_2[index] = *innerIterator_2++;
            indexesBuffer_2[index] = indexes_2[offset_2++];
          }
          if(++outerIterator_2 == keySpans_2.end()) {
            break;
          } else {
            innerIterator_2 = (std::get<Span<U2>>(*outerIterator_2)).begin();
          }
        }

        keysComplete = true;
        keysComplete_2 = true;
      }
    }
  }

  template <typename U1, typename U2>
  inline void processMicroBatch(int microBatchStart, int microBatchSize, int& i, int n,
                                const U1* keys, U1* buffer, const int32_t* indexes,
                                int32_t* indexesBuffer, std::vector<int>& buckets,
                                std::vector<int>& partitions, int& shifts, unsigned int& mask,
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

    if(microBatchSize == tuplesPerHazardCheck &&
       monitor.robustnessIncreaseRequired(microBatchSize)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1;
      mask = numBuckets - 1;

#ifdef ADAPTIVITY_OUTPUT
      std::cout << "RadixBits reduced to " << radixBits << " after tuple " << i
                << " due to reading of ";
      std::cout << (static_cast<float>(microBatchSize) /
                    static_cast<float>(*(eventSet.getCounterDiffsPtr())));
      std::cout << " tuples per TLB store miss" << std::endl;
#endif

      mergePartitions(buffer, indexesBuffer, partitions, buckets, numBuckets, true);
      mergePartitions(buffer_2, indexesBuffer_2, partitions_2, buckets_2, numBuckets, i_2 > 0);

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
                              std::vector<int>& buckets, int numBuckets, bool valuesInBuffer) {
    if(valuesInBuffer) {                    // Skip if no elements have been scattered yet
      for(int j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1];
        auto srcIndex = partitions[j << 1];
        auto numElements = buckets[(j << 1) + 1] - srcIndex;
        std::memcpy(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U));
        std::memcpy(&indexesBuffer[destIndex], &indexesBuffer[srcIndex], numElements * sizeof(U));
      }
    }

    for(int j = 0; j < numBuckets; ++j) { // Merge histogram values
      buckets[j] = buckets[j << 1] + (buckets[(j << 1) + 1] - partitions[j << 1]);
    }

    for(int j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j - 1] = partitions[(j << 1) - 1];
    }
    partitions.resize(numBuckets);
  }

  template <typename U> inline int getMsb(const ExpressionSpanArguments& keySpans, int& n) {
    auto largest = std::numeric_limits<U>::min();
    for(auto& untypedSpan : keySpans) {
      auto& span = std::get<Span<U>>(untypedSpan);
      n += span.size();
      for(auto& key : span) {
        largest = std::max(largest, key);
      }
    }

    int msbToPartition = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartition++;
    }
    return msbToPartition;
  }

  int nInput1;
  const ExpressionSpanArguments& keySpans1;
  std::shared_ptr<T1[]> returnBuffer1;
  std::unique_ptr<T1[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
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
  int tuplesPerHazardCheck;
};

/********************************** UTILITY FUNCTIONS *********************************/

#ifdef DEBUG
template<typename T> void printArray(T* data, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << data[i] << " ";
  } std::cout << std::endl;
}
#endif

template <typename T>
std::pair<std::vector<ExpressionSpanArguments>, std::vector<ExpressionSpanArguments>>
createPartitionsOfSpansAlignedToTableBatches(PartitionedArray<T>& partitionedArray,
                                             const ExpressionSpanArguments& tableKeys) {
  auto keys = partitionedArray.partitionedKeys.get();
  auto indexes = partitionedArray.indexes.get();
  auto partitions = *(partitionedArray.partitionPositions.get());

#ifdef DEBUG
  if(partitions.size() > 0) {
    int size = partitions.back().first + partitions.back().second;
    std::cout << "Keys: ";
    printArray<T>(keys, size);
    std::cout << "Indexes: ";
    printArray<int32_t>(indexes, size);
    std::cout << "Partitions: ";
    for(auto& pair : partitions) {
      std::cout << "[" << pair.first << "," << pair.second << "] ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "No output partitions" << std::endl;
  }
#endif

  std::vector<ExpressionSpanArguments> partitionsOfKeySpans, partitionsOfIndexSpans;
  partitionsOfKeySpans.reserve(partitions.size());
  partitionsOfIndexSpans.reserve(partitions.size());

  int cumulativeBatchSize = 0;
  std::vector<int> cumulativeBatchSizes(1, 0);
  cumulativeBatchSizes.reserve(tableKeys.size() + 1);
  for(const auto& batch : tableKeys) {
    cumulativeBatchSize += get<Span<T>>(batch).size();
    cumulativeBatchSizes.push_back(cumulativeBatchSize);
  }

  for(size_t partitionNum = 0; partitionNum < partitions.size(); ++partitionNum) {
    int index = partitions[partitionNum].first;
    auto partitionEndIndex = index + partitions[partitionNum].second;
    ExpressionSpanArguments outputKeySpans, outputIndexSpans;
    outputKeySpans.reserve(tableKeys.size());
    outputIndexSpans.reserve(tableKeys.size());
    int prevSpanEndIndex = partitions[partitionNum].first;
    for(size_t batchNum = 1; batchNum < cumulativeBatchSizes.size(); ++batchNum) {
      while(indexes[index] < cumulativeBatchSizes[batchNum] && index < partitionEndIndex) {
        indexes[index++] -= cumulativeBatchSizes[batchNum - 1];
      }
      if(index == prevSpanEndIndex) {
        outputKeySpans.emplace_back(Span<T>());
        outputIndexSpans.emplace_back(Span<int32_t>());
      } else {
        outputKeySpans.emplace_back(Span<T>(keys + prevSpanEndIndex, index - prevSpanEndIndex,
                                            [ptr = partitionedArray.partitionedKeys]() {}));
        outputIndexSpans.emplace_back(Span<int32_t>(indexes + prevSpanEndIndex,
                                                    index - prevSpanEndIndex,
                                                    [ptr = partitionedArray.indexes]() {}));
        prevSpanEndIndex = index;
      }
    }
    partitionsOfKeySpans.push_back(std::move(outputKeySpans));
    partitionsOfIndexSpans.push_back(std::move(outputIndexSpans));
  }

  return std::make_pair(std::move(partitionsOfKeySpans), std::move(partitionsOfIndexSpans));
}

/*********************************** ENTRY FUNCTION ***********************************/

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           const ExpressionSpanArguments& tableOneKeys,
                                           const ExpressionSpanArguments& tableTwoKeys) {
  static_assert(std::is_integral<T1>::value, "PartitionOperators column must be an integer type");
  static_assert(std::is_integral<T2>::value, "PartitionOperators column must be an integer type");

  TwoPartitionedArrays partitionedTables = [partitionImplementation, &tableOneKeys,
                                            &tableTwoKeys]() {
    if(partitionImplementation == PartitionOperators::RadixBitsFixed) {
      auto partitionOperator = Partition<T1, T2>(tableOneKeys, tableTwoKeys);
      return partitionOperator.processInput();
    } else if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
      auto partitionOperator = PartitionAdaptive<T1, T2>(tableOneKeys, tableTwoKeys);
      return partitionOperator.processInput();
    } else {
      throw std::runtime_error("Invalid selection of 'Partition' implementation!");
    }
  }();

  auto [tableOnePartitionsOfKeySpans, tableOnePartitionsOfIndexSpans] =
      createPartitionsOfSpansAlignedToTableBatches<T1>(partitionedTables.partitionedArrayOne,
                                                       tableOneKeys);
  auto [tableTwoPartitionsOfKeySpans, tableTwoPartitionsOfIndexSpans] =
      createPartitionsOfSpansAlignedToTableBatches<T2>(partitionedTables.partitionedArrayTwo,
                                                       tableTwoKeys);
  return {std::move(tableOnePartitionsOfKeySpans), std::move(tableOnePartitionsOfIndexSpans),
          std::move(tableTwoPartitionsOfKeySpans), std::move(tableTwoPartitionsOfIndexSpans)};
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
