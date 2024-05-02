#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include "constants/machineConstants.hpp"
#include "utilities/dataStructures.hpp"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/systemInformation.hpp"

#define USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1
// #define CREATE_SPANS_ALIGNED_TO_BATCHES
#define ADAPTIVITY_OUTPUT
// #define DEBUG
// #define CHANGE_PARTITION_TO_SORT_FOR_TESTING

#ifndef CREATE_SPANS_ALIGNED_TO_BATCHES
// #define INCLUDE_MIN_PARTITION_SIZE // Toggle, requires not creating spans aligned to batches
#endif

#ifdef INCLUDE_MIN_PARTITION_SIZE
constexpr int MIN_PARTITION_SIZE = 1;
#endif

constexpr int TUPLES_PER_HAZARD_CHECK = 10 * 1000;

namespace adaptive {

/****************************** FORWARD DECLARATIONS ******************************/

#ifdef DEBUG
template <typename T> void printArray(T* data, int size) {
  for(int i = 0; i < size; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}
#endif

class MonitorPartition {
public:
  explicit MonitorPartition(const long_long* sTlbStoreMisses_)
      : sTlbStoreMisses(sTlbStoreMisses_), tuplesPerTlbStoreMiss(10.0) {}
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
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / sizeof(T1);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared_for_overwrite<int32_t[]>(nInput1);
    tmpIndexes1 = std::make_unique_for_overwrite<int32_t[]>(nInput1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < nInput1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared_for_overwrite<int32_t[]>(nInput2);
    tmpIndexes2 = std::make_unique_for_overwrite<int32_t[]>(nInput2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < nInput2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>> processInput() {
    performPartition();
    return TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>>{
        PartitionedArray<std::remove_cv_t<T1>>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<std::remove_cv_t<T2>>{
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

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartition == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      for(size_t j = 0; j < partitions1.size(); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
        prevPartitionEnd1 = partitions1[j];
        prevPartitionEnd2 = partitions2[j];
      }
      return;
    }

    for(size_t j = 0; j < partitions1.size(); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          if(tmpBuffer1 == nullptr) {
            // Lazily allocate tmpBuffer
            tmpBuffer1 = std::make_unique_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
            tmpBuffer2 = std::make_unique_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
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

  inline void performPartitionAux(int n1, std::remove_cv_t<T1>* keys1,
                                  std::remove_cv_t<T1>* buffer1, int32_t* indexes1,
                                  int32_t* indexesBuffer1, int offset1, int n2,
                                  std::remove_cv_t<T2>* keys2, std::remove_cv_t<T2>* buffer2,
                                  int32_t* indexes2, int32_t* indexesBuffer2, int offset2,
                                  int msbToPartition, int radixBits, bool copyRequired) {
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

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartition == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      if(copyRequired) {
        std::memcpy(keys1, buffer1, n1 * sizeof(T1));
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int32_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int32_t));
      }
      for(size_t j = 0; j < partitions1.size(); ++j) {
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

    for(size_t j = 0; j < partitions1.size(); ++j) {
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
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
  std::shared_ptr<int32_t[]> returnIndexes2;
  std::unique_ptr<int32_t[]> tmpIndexes2;
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;

  int radixBitsOperator;
  int msbToPartitionInput;
  int maxElementsPerPartition;
};

/****************************** SINGLE-THREADED ******************************/

template <typename T1, typename T2> class PartitionAdaptive {
public:
  PartitionAdaptive(const ExpressionSpanArguments& keySpans1_,
                    const ExpressionSpanArguments& keySpans2_)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_),
        eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(TUPLES_PER_HAZARD_CHECK) {
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
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / sizeof(T1);
#endif

    buckets1 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed

    returnIndexes1 = std::make_shared_for_overwrite<int32_t[]>(nInput1);
    tmpIndexes1 = std::make_unique_for_overwrite<int32_t[]>(nInput1);

    auto indexesPtr1 = tmpIndexes1.get();
    for(auto i = 0; i < nInput1; ++i) {
      indexesPtr1[i] = i;
    }

    buckets2 = std::vector<int>(1 + (1 << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed

    returnIndexes2 = std::make_shared_for_overwrite<int32_t[]>(nInput2);
    tmpIndexes2 = std::make_unique_for_overwrite<int32_t[]>(nInput2);

    auto indexesPtr2 = tmpIndexes2.get();
    for(auto i = 0; i < nInput2; ++i) {
      indexesPtr2[i] = i;
    }
  }

  TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>> processInput() {
    performPartition();
    return TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>>{
        PartitionedArray<std::remove_cv_t<T1>>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<std::remove_cv_t<T2>>{
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
    int startingNumBuckets = numBuckets;
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

    std::fill(buckets1.begin(), buckets1.begin() + startingNumBuckets + 1, 0);
    std::fill(buckets2.begin(), buckets2.begin() + startingNumBuckets + 1, 0);

    msbToPartition -= radixBits;

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartition == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      for(size_t j = 0; j < partitions1.size(); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
        prevPartitionEnd1 = partitions1[j];
        prevPartitionEnd2 = partitions2[j];
      }
      return;
    }

    for(size_t j = 0; j < partitions1.size(); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          if(tmpBuffer1 == nullptr) {
            // Lazily allocate tmpBuffer
            tmpBuffer1 = std::make_unique_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
            tmpBuffer2 = std::make_unique_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
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

  inline void performPartitionAux(int n1, std::remove_cv_t<T1>* keys1,
                                  std::remove_cv_t<T1>* buffer1, int32_t* indexes1,
                                  int32_t* indexesBuffer1, int offset1, int n2,
                                  std::remove_cv_t<T2>* keys2, std::remove_cv_t<T2>* buffer2,
                                  int32_t* indexes2, int32_t* indexesBuffer2, int offset2,
                                  int msbToPartition, int radixBits, bool copyRequired) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    int startingNumBuckets = numBuckets;
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

    std::fill(buckets1.begin(), buckets1.begin() + startingNumBuckets + 1, 0);
    std::fill(buckets2.begin(), buckets2.begin() + startingNumBuckets + 1, 0);

    msbToPartition -= radixBits;

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartition == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      if(copyRequired) {
        std::memcpy(keys1, buffer1, n1 * sizeof(T1));
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int32_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int32_t));
      }
      for(size_t j = 0; j < partitions1.size(); ++j) {
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

    for(size_t j = 0; j < partitions1.size(); ++j) {
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

  template <typename U1, typename U2, typename OuterIt, typename InnerIt, typename OuterIt_2,
            typename InnerIt_2>
  inline void processSpansMicroBatch(
      OuterIt& outerIterator, InnerIt& innerIterator, const ExpressionSpanArguments& keySpans,
      std::remove_cv_t<U1>* buffer, const int32_t* indexes, int32_t* indexesBuffer,
      std::vector<int>& buckets, std::vector<int>& partitions, int& offset, bool& keysComplete,
      int& shifts, unsigned int& mask, int& radixBits, int& numBuckets, OuterIt_2& outerIterator_2,
      InnerIt_2& innerIterator_2, const ExpressionSpanArguments& keySpans_2,
      std::remove_cv_t<U2>* buffer_2, const int32_t* indexes_2, int32_t* indexesBuffer_2,
      std::vector<int>& buckets_2, std::vector<int>& partitions_2, int& offset_2,
      bool& keysComplete_2) {
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

    if(processed == tuplesPerHazardCheck && radixBits > minimumRadixBits &&
       monitor.robustnessIncreaseRequired(tuplesPerHazardCheck)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1;
      mask = numBuckets - 1;

#ifdef ADAPTIVITY_OUTPUT
      std::cout << "RadixBits reduced to " << radixBits << " due to reading of ";
      std::cout << (static_cast<float>(tuplesPerHazardCheck) /
                    static_cast<float>(
                        *(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)));
      std::cout << " tuples per TLB store miss" << std::endl;
#endif

      mergePartitions(buffer, indexesBuffer, partitions, buckets, numBuckets, true);
      mergePartitions(buffer_2, indexesBuffer_2, partitions_2, buckets_2, numBuckets, offset_2 > 0);
    }

    if(radixBits <= minimumRadixBits) { // Complete partitioning to avoid unnecessary checks
      if(outerIterator != keySpans.end()) {
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
      }

      if(outerIterator_2 != keySpans_2.end()) {
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
      }

      keysComplete = true;
      keysComplete_2 = true;
    }
  }

  template <typename U1, typename U2>
  inline void processMicroBatch(int microBatchStart, int microBatchSize, int& i, int n,
                                const U1* keys, std::remove_cv_t<U1>* buffer,
                                const int32_t* indexes, int32_t* indexesBuffer,
                                std::vector<int>& buckets, std::vector<int>& partitions,
                                int& shifts, unsigned int& mask, int& radixBits, int& numBuckets,
                                int& i_2, int n_2, const U2* keys_2, std::remove_cv_t<U2>* buffer_2,
                                const int32_t* indexes_2, int32_t* indexesBuffer_2,
                                std::vector<int>& buckets_2, std::vector<int>& partitions_2) {
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
                    static_cast<float>(
                        *(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)));
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
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int32_t));
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
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
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

/************************** SINGLE-THREADED FOR MULTI-THREADED **************************/

struct TwoPartitionedArraysPartitionsOnly {
  std::unique_ptr<vectorOfPairs<int, int>> tableOnePartitionPositions;
  std::unique_ptr<vectorOfPairs<int, int>> tableTwoPartitionPositions;
};

template <typename T> struct PartitionedArrayAlt {
  std::shared_ptr<T[]> partitionedKeys;
  std::shared_ptr<int32_t[]> indexes;
  std::unique_ptr<std::vector<int>> partitionPositions;
};

template <typename T1, typename T2> struct TwoPartitionedArraysAlt {
  PartitionedArrayAlt<T1> partitionedArrayOne;
  PartitionedArrayAlt<T2> partitionedArrayTwo;
};

template <typename T1, typename T2> class PartitionAdaptiveRawArrays {
public:
  PartitionAdaptiveRawArrays()
      : eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(TUPLES_PER_HAZARD_CHECK) {

    size_t initialSize = 2 * l2cacheSize() / 4;
    keysTmpBuffer1.resize(initialSize);
    indexesTmpBuffer1.resize(initialSize);
    keysTmpBuffer2.resize(initialSize);
    indexesTmpBuffer2.resize(initialSize);
  }

  TwoPartitionedArraysPartitionsOnly
  processInput(int n1, T1* keys1, int32_t* indexes1, int overallOffset1_, int n2, T2* keys2,
               int32_t* indexes2, int overallOffset2_, int minimumRadixBits_, int radixBits,
               int msbToPartitionInput_, int maxElementsPerPartition_,
               std::atomic<int>* totalOutputPartitions_) {
    nInput1 = n1;
    keysInput1 = keys1;
    indexesInput1 = indexes1;
    overallOffset1 = overallOffset1_;

    nInput2 = n2;
    keysInput2 = keys2;
    indexesInput2 = indexes2;
    overallOffset2 = overallOffset2_;

    minimumRadixBits = minimumRadixBits_;
    radixBitsOperator = radixBits;
    msbToPartitionInput = msbToPartitionInput_;
    maxElementsPerPartition = maxElementsPerPartition_;
    totalOutputPartitions = totalOutputPartitions_;

    if(static_cast<size_t>(nInput1) > keysTmpBuffer1.size()) {
      keysTmpBuffer1.resize(nInput1);
      indexesTmpBuffer1.resize(nInput1);
    }
    if(static_cast<size_t>(nInput2) > keysTmpBuffer2.size()) {
      keysTmpBuffer2.resize(nInput2);
      indexesTmpBuffer2.resize(nInput2);
    }

    buckets1.resize(1 + (1 << radixBitsOperator));
    std::fill(buckets1.begin(), buckets1.end(), 0);
    buckets2.resize(1 + (1 << radixBitsOperator));
    std::fill(buckets1.begin(), buckets1.end(), 0);

    performPartition(nInput1, keysInput1, keysTmpBuffer1.data(), indexesInput1,
                     indexesTmpBuffer1.data(), overallOffset1, nInput2, keysInput2,
                     keysTmpBuffer2.data(), indexesInput2, indexesTmpBuffer2.data(), overallOffset2,
                     msbToPartitionInput, radixBitsOperator, true);
    totalOutputPartitions->fetch_add(outputPartitions1.size());
    return TwoPartitionedArraysPartitionsOnly{
        std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1)),
        std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))};
  }

private:
  inline void performPartition(int n1, T1* keys1, T1* buffer1, int32_t* indexes1,
                               int32_t* indexesBuffer1, int offset1, int n2, T2* keys2, T2* buffer2,
                               int32_t* indexes2, int32_t* indexesBuffer2, int offset2,
                               int msbToPartition, int radixBits, bool copyRequired) {
    radixBits = std::min(msbToPartition, radixBits);
    int shifts = msbToPartition - radixBits;
    int numBuckets = 1 << radixBits;
    int startingNumBuckets = numBuckets;
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

    std::fill(buckets1.begin(), buckets1.begin() + startingNumBuckets + 1, 0);
    std::fill(buckets2.begin(), buckets2.begin() + startingNumBuckets + 1, 0);

    msbToPartition -= radixBits;

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartition == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      if(copyRequired) {
        std::memcpy(keys1, buffer1, n1 * sizeof(T1));
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int32_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int32_t));
      }
      for(size_t j = 0; j < partitions1.size(); ++j) {
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

    for(size_t j = 0; j < partitions1.size(); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          performPartition(partitions1[j] - prevPartitionEnd1, buffer1 + prevPartitionEnd1,
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
                    static_cast<float>(
                        *(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)));
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
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int32_t));
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

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  int tuplesPerHazardCheck;
  std::vector<T1> keysTmpBuffer1;
  std::vector<int32_t> indexesTmpBuffer1;
  std::vector<T2> keysTmpBuffer2;
  std::vector<int32_t> indexesTmpBuffer2;

  int nInput1{};
  T1* keysInput1{};
  int32_t* indexesInput1{};
  std::vector<int> buckets1{};
  vectorOfPairs<int, int> outputPartitions1{};
  int overallOffset1{};

  int nInput2{};
  T2* keysInput2{};
  int32_t* indexesInput2{};
  std::vector<int> buckets2{};
  vectorOfPairs<int, int> outputPartitions2{};
  int overallOffset2{};

  int minimumRadixBits{};
  int radixBitsOperator{};
  int msbToPartitionInput{};
  int maxElementsPerPartition{};
  std::atomic<int>* totalOutputPartitions{};
};

template <typename T1, typename T2>
static PartitionAdaptiveRawArrays<T1, T2>& getPartitionAdaptiveRawArrays() {
  thread_local static PartitionAdaptiveRawArrays<T1, T2> partitionAdaptiveRawArrays;
  return partitionAdaptiveRawArrays;
}

/************************************ MULTI-THREADED ***********************************/

template <typename T1, typename T2> class PartitionAdaptiveParallelFirstPass {
public:
  PartitionAdaptiveParallelFirstPass(const ExpressionSpanArguments& keySpans1_, int overallOffset1,
                                     const ExpressionSpanArguments& keySpans2_, int overallOffset2,
                                     int batchNumStart1_, int batchNumEnd1_, int n1,
                                     int batchNumStart2_, int batchNumEnd2_, int n2,
                                     int minimumRadixBits_, int startingRadixBits,
                                     std::atomic<int>& globalRadixBits_, int msbToPartitionInput_,
                                     int maxElementsPerPartition_,
                                     std::atomic<int>& threadsStillRunning_)
      : nInput1(n1), keySpans1(keySpans1_), batchNumStart1(batchNumStart1_),
        batchNumEnd1(batchNumEnd1_), nInput2(n2), keySpans2(keySpans2_),
        batchNumStart2(batchNumStart2_), batchNumEnd2(batchNumEnd2_),
        minimumRadixBits(minimumRadixBits_), radixBits(startingRadixBits),
        globalRadixBits(globalRadixBits_), msbToPartitionInput(msbToPartitionInput_),
        maxElementsPerPartition(maxElementsPerPartition_),
        threadsStillRunning(threadsStillRunning_), eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(TUPLES_PER_HAZARD_CHECK) {

    buckets1 = std::vector<int>(1 + (1 << radixBits), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
    returnIndexes1 = std::make_shared_for_overwrite<int32_t[]>(nInput1);
    inputIndexes1 = std::make_unique_for_overwrite<int32_t[]>(nInput1);

#ifdef DEBUG
    std::cout << "For this thread: overallOffset1: " << overallOffset1
              << ", overallOffset2: " << overallOffset2 << ", n1: " << nInput1
              << ", n2: " << nInput2 << std::endl;
#endif

    auto indexesPtr1 = inputIndexes1.get();
    for(auto i = 0; i < nInput1; ++i) {
      indexesPtr1[i] = overallOffset1 + i;
    }

    buckets2 = std::vector<int>(1 + (1 << radixBits), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    returnIndexes2 = std::make_shared_for_overwrite<int32_t[]>(nInput2);
    inputIndexes2 = std::make_unique_for_overwrite<int32_t[]>(nInput2);

    auto indexesPtr2 = inputIndexes2.get();
    for(auto i = 0; i < nInput2; ++i) {
      indexesPtr2[i] = overallOffset2 + i;
    }

#ifdef DEBUG
    std::cout << "Initial indexes1: ";
    printArray<int32_t>(indexesPtr1, nInput1);
    std::cout << "Initial indexes2: ";
    printArray<int32_t>(indexesPtr2, nInput2);
#endif
  }

  TwoPartitionedArraysAlt<std::remove_cv_t<T1>, std::remove_cv_t<T2>> processInput() {
    performPartition();

#ifdef DEBUG
    std::cout << "Final radix bits value for this thread: " << radixBits << std::endl;

    std::cout << "This thread has completed first pass partitioning: " << std::endl;
    std::cout << "Table 1 results: " << std::endl;
    std::cout << "partitions1.size(): " << partitions1.size() << std::endl;
    std::cout << "partition1Positions: " << std::endl;
    printArray<int32_t>(partitions1.data(), partitions1.size());
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T1>>(returnBuffer1.get(), nInput1);
    std::cout << "Indexes: " << std::endl;
    printArray<int32_t>(returnIndexes1.get(), nInput1);

    std::cout << "Table 2 results: " << std::endl;
    std::cout << "partitions2.size(): " << partitions2.size() << std::endl;
    std::cout << "partition2Positions: " << std::endl;
    printArray<int32_t>(partitions2.data(), partitions2.size());
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T2>>(returnBuffer2.get(), nInput2);
    std::cout << "Indexes: " << std::endl;
    printArray<int32_t>(returnIndexes2.get(), nInput2);
#endif

    return TwoPartitionedArraysAlt<std::remove_cv_t<T1>, std::remove_cv_t<T2>>{
        PartitionedArrayAlt<std::remove_cv_t<T1>>{
            returnBuffer1, returnIndexes1,
            std::make_unique<std::vector<int>>(std::move(partitions1))},
        PartitionedArrayAlt<std::remove_cv_t<T2>>{
            returnBuffer2, returnIndexes2,
            std::make_unique<std::vector<int>>(std::move(partitions2))}};
  }

private:
  inline void performPartition() {
    auto buffer1 = returnBuffer1.get();
    auto indexes1 = inputIndexes1.get();
    auto indexesBuffer1 = returnIndexes1.get();
    auto buffer2 = returnBuffer2.get();
    auto indexes2 = inputIndexes2.get();
    auto indexesBuffer2 = returnIndexes2.get();

    int shifts = msbToPartitionInput - radixBits;
    int numBuckets = 1 << radixBits;
    unsigned int mask = numBuckets - 1;

    // Initialise span iterators
    auto keySpansStart1 = keySpans1.begin();
    std::advance(keySpansStart1, batchNumStart1);
    auto keySpansEnd1 = keySpans1.begin();
    std::advance(keySpansEnd1, batchNumEnd1);
    auto keySpansStart2 = keySpans2.begin();
    std::advance(keySpansStart2, batchNumStart2);
    auto keySpansEnd2 = keySpans2.begin();
    std::advance(keySpansEnd2, batchNumEnd2);

#ifdef DEBUG
    std::cout << "batchNumStart1: " << batchNumStart1 << ", batchNumEnd1: " << batchNumEnd1
              << ", batchNumStart2: " << batchNumStart2 << ", batchNumEnd2: " << batchNumEnd2
              << std::endl;
#endif

    // Complete histogram for array 1
    int i;
    for(auto spansIt = keySpansStart1; spansIt != keySpansEnd1; ++spansIt) {
      for(auto& key : std::get<Span<T1>>(*spansIt)) {
        buckets1[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    partitions1 = buckets1;

    // Complete histogram for array 2
    for(auto spansIt = keySpansStart2; spansIt != keySpansEnd2; ++spansIt) {
      for(auto& key : std::get<Span<T2>>(*spansIt)) {
        buckets2[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    partitions2 = buckets2;

    bool keysComplete1 = false, keysComplete2 = false;
    auto outerIterator1 = keySpansStart1;
    auto innerIterator1 = (std::get<Span<T1>>(*outerIterator1)).begin();
    auto outerIterator2 = keySpansStart2;
    auto innerIterator2 = (std::get<Span<T2>>(*outerIterator2)).begin();
    int offset1 = 0, offset2 = 0;

    while(!keysComplete1 || !keysComplete2) { // NOLINT
      if(!keysComplete1) {                    // NOLINT
        processSpansMicroBatch<T1, T2>(
            outerIterator1, keySpansEnd1, innerIterator1, buffer1, indexes1, indexesBuffer1,
            buckets1, partitions1, offset1, keysComplete1, shifts, mask, numBuckets, outerIterator2,
            keySpansEnd2, innerIterator2, buffer2, indexes2, indexesBuffer2, buckets2, partitions2,
            offset2, keysComplete2);
      }
      if(!keysComplete2) { // NOLINT
        processSpansMicroBatch<T2, T1>(
            outerIterator2, keySpansEnd2, innerIterator2, buffer2, indexes2, indexesBuffer2,
            buckets2, partitions2, offset2, keysComplete2, shifts, mask, numBuckets, outerIterator1,
            keySpansEnd1, innerIterator1, buffer1, indexes1, indexesBuffer1, buckets1, partitions1,
            offset1, keysComplete1);
      }
    }

    threadsStillRunning.fetch_sub(1);
    if(radixBits > minimumRadixBits) { // Final synchronisation required with all complete threads
      while(threadsStillRunning > 0) {
        /* busy wait */
      }
      int finalGlobalRadixBits = globalRadixBits.load();
      while(radixBits != finalGlobalRadixBits) {
        --radixBits;
        numBuckets >>= 1;
        mergePartitions(buffer1, indexesBuffer1, partitions1, buckets1, numBuckets, true);
        mergePartitions(buffer2, indexesBuffer2, partitions2, buckets2, numBuckets, true);
      }
    }
  }

  template <typename U1, typename U2, typename OuterIt, typename InnerIt, typename OuterIt_2,
            typename InnerIt_2>
  inline void processSpansMicroBatch(
      OuterIt& outerIterator, OuterIt& outerIteratorEnd, InnerIt& innerIterator,
      std::remove_cv_t<U1>* buffer, const int32_t* indexes, int32_t* indexesBuffer,
      std::vector<int>& buckets, std::vector<int>& partitions, int& offset, bool& keysComplete,
      int& shifts, unsigned int& mask, int& numBuckets, OuterIt_2& outerIterator_2,
      OuterIt_2& outerIteratorEnd_2, InnerIt_2& innerIterator_2, std::remove_cv_t<U2>* buffer_2,
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
        if(++outerIterator == outerIteratorEnd) {
          keysComplete = true;
          break;
        } else {
          innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
        }
      }
    }
    eventSet.readCountersAndUpdateDiff();
    offset += processed;

    int tmpGlobalRadixBits;
    if(processed == tuplesPerHazardCheck && radixBits > minimumRadixBits &&
       monitor.robustnessIncreaseRequired(tuplesPerHazardCheck)) { // Local adjustment required
      if(globalRadixBits.load() > radixBits - 1) {                 // Global adjustment required
        globalRadixBits.store(radixBits - 1); // Sync global radix bits to new local value
        tmpGlobalRadixBits = radixBits - 1;
#ifdef ADAPTIVITY_OUTPUT
        std::cout << "Global radixBits reduced to " << tmpGlobalRadixBits << " due to reading of ";
        std::cout << (static_cast<float>(tuplesPerHazardCheck) /
                      static_cast<float>(
                          *(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)));
        std::cout << " tuples per TLB store miss" << std::endl;
#endif
      } else {
        tmpGlobalRadixBits = globalRadixBits;
      }
    } else {
      tmpGlobalRadixBits = globalRadixBits;
    }

    while(radixBits > tmpGlobalRadixBits) { // Sync with global
      --radixBits;
      ++shifts;
      numBuckets >>= 1;
      mask = numBuckets - 1;

      mergePartitions(buffer, indexesBuffer, partitions, buckets, numBuckets, true);
      mergePartitions(buffer_2, indexesBuffer_2, partitions_2, buckets_2, numBuckets, offset_2 > 0);

      if(radixBits <= minimumRadixBits) { // Complete partitioning to avoid unnecessary checks
        if(outerIterator != outerIteratorEnd) {
          while(true) {
            while(innerIterator != (std::get<Span<U1>>(*outerIterator)).end()) {
              auto index = buckets[((*innerIterator) >> shifts) & mask]++;
              buffer[index] = *(innerIterator++);
              indexesBuffer[index] = indexes[offset++];
            }
            if(++outerIterator == outerIteratorEnd) {
              break;
            } else {
              innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
            }
          }
        }

        if(outerIterator_2 != outerIteratorEnd_2) {
          while(true) {
            while(innerIterator_2 != (std::get<Span<U2>>(*outerIterator_2)).end()) {
              auto index = buckets_2[((*innerIterator_2) >> shifts) & mask]++;
              buffer_2[index] = *innerIterator_2++;
              indexesBuffer_2[index] = indexes_2[offset_2++];
            }
            if(++outerIterator_2 == outerIteratorEnd_2) {
              break;
            } else {
              innerIterator_2 = (std::get<Span<U2>>(*outerIterator_2)).begin();
            }
          }
        }

        keysComplete = true;
        keysComplete_2 = true;
        break;
      }
    }
  }

  template <typename U>
  inline void mergePartitions(U* buffer, int32_t* indexesBuffer, std::vector<int>& partitions,
                              std::vector<int>& buckets, int numBuckets, bool valuesInBuffer) {
    if(valuesInBuffer) {                    // Skip if no elements have been scattered yet
      for(int j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1];
        auto srcIndex = partitions[(j << 1) + 1];
        auto numElements = buckets[(j << 1) + 1] - srcIndex;
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int32_t));
      }
    }

    for(int j = 0; j < numBuckets; ++j) { // Merge histogram values
      buckets[j] = buckets[j << 1] + (buckets[(j << 1) + 1] - partitions[(j << 1) + 1]);
    }

    for(int j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j] = partitions[j << 1];
    }
    partitions.resize(numBuckets + 1);
  }

  int nInput1;
  const ExpressionSpanArguments& keySpans1;
  int batchNumStart1;
  int batchNumEnd1;
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  std::unique_ptr<int32_t[]> inputIndexes1;
  std::vector<int> buckets1;
  std::vector<int> partitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
  int batchNumStart2;
  int batchNumEnd2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
  std::shared_ptr<int32_t[]> returnIndexes2;
  std::unique_ptr<int32_t[]> inputIndexes2;
  std::vector<int> buckets2;
  std::vector<int> partitions2;

  int minimumRadixBits;
  int radixBits;
  std::atomic<int>& globalRadixBits;
  int msbToPartitionInput;
  int maxElementsPerPartition;
  std::atomic<int>& threadsStillRunning;

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  int tuplesPerHazardCheck;
};

template <typename T1, typename T2> class PartitionAdaptiveParallel {
public:
  PartitionAdaptiveParallel(const ExpressionSpanArguments& keySpans1_,
                            const ExpressionSpanArguments& keySpans2_, int dop_)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_), dop(dop_),
        threadPool(ThreadPool::getInstance()), synchroniser(Synchroniser::getInstance()),
        totalOutputPartitions(0) {

    std::string startName = "Partition_startRadixBits";
    startingRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(startName));

    std::string minName = "Partition_minRadixBits";
    minimumRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));

    if(dop == 1) {
      msbToPartitionInput =
          std::max(getMsb<T1>(keySpans1, nInput1), getMsb<T2>(keySpans2, nInput2));
    } else {
      int msb1, msb2;
      threadPool.enqueue([this, &msb1] {
        msb1 = getMsb<T1>(keySpans1, nInput1);
        synchroniser.taskComplete();
      });
      threadPool.enqueue([this, &msb2] {
        msb2 = getMsb<T2>(keySpans2, nInput2);
        synchroniser.taskComplete();
      });
      synchroniser.waitUntilComplete(2);
      msbToPartitionInput = std::max(msb1, msb2);
    }

    startingRadixBits = std::min(msbToPartitionInput, startingRadixBits);
    globalRadixBits = startingRadixBits;

#ifdef CHANGE_PARTITION_TO_SORT_FOR_TESTING
    maxElementsPerPartition = 1;
#else
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / sizeof(T1);
#endif

    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1);
    returnIndexes1 = std::make_shared_for_overwrite<int32_t[]>(nInput1);

    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    returnIndexes2 = std::make_shared_for_overwrite<int32_t[]>(nInput2);
  }

  TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>> processInput() {
    performPartition();
    return TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>>{
        PartitionedArray<std::remove_cv_t<T1>>{
            returnBuffer1, returnIndexes1,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions1))},
        PartitionedArray<std::remove_cv_t<T2>>{
            returnBuffer2, returnIndexes2,
            std::make_unique<vectorOfPairs<int, int>>(std::move(outputPartitions2))}};
  }

private:
  inline void performPartition() {
    auto cumulativeBatchSizes1 = createCumulativeBatchSizes<T1>(keySpans1);
    auto cumulativeBatchSizes2 = createCumulativeBatchSizes<T2>(keySpans2);

#ifdef DEBUG
    std::cout << "cumulativeBatchSizes1: " << std::endl;
    printArray<int>(cumulativeBatchSizes1.data(), cumulativeBatchSizes1.size());
    std::cout << "cumulativeBatchSizes2: " << std::endl;
    printArray<int>(cumulativeBatchSizes2.data(), cumulativeBatchSizes2.size());
#endif

    int numBatches1 = keySpans1.size();
    int numBatches2 = keySpans2.size();
    int firstPassDop = std::min(std::min(numBatches1, numBatches2), dop);
    threadsStillRunning = firstPassDop;

#ifdef DEBUG
    std::cout << "firstPassDop: " << firstPassDop << std::endl;
    std::cout << "numBatches1: " << numBatches1 << std::endl;
    std::cout << "numBatches2: " << numBatches2 << std::endl;
#endif

    int baselineBatchesPerThread1 = numBatches1 / firstPassDop;
    int baselineBatchesPerThread2 = numBatches2 / firstPassDop;
    int remainingBatches1 = numBatches1 % firstPassDop;
    int remainingBatches2 = numBatches2 % firstPassDop;
    int startBatchNum1 = 0, startBatchNum2 = 0;
    int endBatchNum1, endBatchNum2;
    int batchesPerThread1, batchesPerThread2;
    int overallOffset1, overallOffset2;
    int elementsInThread1, elementsInThread2;

    std::vector<PartitionedArrayAlt<std::remove_cv_t<T1>>> firstPassPartitions1(firstPassDop);
    std::vector<PartitionedArrayAlt<std::remove_cv_t<T2>>> firstPassPartitions2(firstPassDop);

    for(auto taskNum = 0; taskNum < firstPassDop; ++taskNum) {
      if(taskNum < firstPassDop - 1) {
        batchesPerThread1 = baselineBatchesPerThread1 + (taskNum < remainingBatches1);
        batchesPerThread2 = baselineBatchesPerThread2 + (taskNum < remainingBatches2);
      } else {
        batchesPerThread1 = numBatches1 - startBatchNum1;
        batchesPerThread2 = numBatches2 - startBatchNum2;
      }

      overallOffset1 = cumulativeBatchSizes1[startBatchNum1];
      overallOffset2 = cumulativeBatchSizes2[startBatchNum2];

      endBatchNum1 = startBatchNum1 + batchesPerThread1;
      endBatchNum2 = startBatchNum2 + batchesPerThread2;

      elementsInThread1 = cumulativeBatchSizes1[endBatchNum1] - overallOffset1;
      elementsInThread2 = cumulativeBatchSizes2[endBatchNum2] - overallOffset2;

      threadPool.enqueue([this, overallOffset1, overallOffset2, startBatchNum1, endBatchNum1,
                          elementsInThread1, startBatchNum2, endBatchNum2, elementsInThread2,
                          taskNum, &firstPassPartitions1, &firstPassPartitions2] {
        auto op = PartitionAdaptiveParallelFirstPass<std::remove_cv_t<T1>, std::remove_cv_t<T2>>(
            keySpans1, overallOffset1, keySpans2, overallOffset2, startBatchNum1, endBatchNum1,
            elementsInThread1, startBatchNum2, endBatchNum2, elementsInThread2, minimumRadixBits,
            startingRadixBits, globalRadixBits, msbToPartitionInput, maxElementsPerPartition,
            threadsStillRunning);
        auto results = op.processInput();
        {
          std::lock_guard<std::mutex> lock(resultsMutex);
          firstPassPartitions1[taskNum] = std::move(results.partitionedArrayOne);
          firstPassPartitions2[taskNum] = std::move(results.partitionedArrayTwo);
        }
        synchroniser.taskComplete();
      });

      startBatchNum1 += batchesPerThread1;
      startBatchNum2 += batchesPerThread2;
    }

    synchroniser.waitUntilComplete(firstPassDop);

#ifdef DEBUG
    std::cout << "Before updating: "
              << "GlobalRadixBits: " << globalRadixBits
              << ", msbToPartitionInput: " << msbToPartitionInput << std::endl;
#endif

    msbToPartitionInput -= globalRadixBits;
    globalRadixBits = std::min(msbToPartitionInput, globalRadixBits.load());

#ifdef DEBUG
    std::cout << "After updating: "
              << "GlobalRadixBits: " << globalRadixBits
              << ", msbToPartitionInput: " << msbToPartitionInput << std::endl;
#endif

    std::vector<int> partitions1, partitions2;
    if(dop == 1) {
      partitions1 = mergeAndCreatePartitionsVec<std::remove_cv_t<T1>>(
          firstPassPartitions1, returnBuffer1.get(), returnIndexes1.get());
      partitions2 = mergeAndCreatePartitionsVec<std::remove_cv_t<T2>>(
          firstPassPartitions2, returnBuffer2.get(), returnIndexes2.get());
    } else if(dop < 4) {
      threadPool.enqueue([this, &partitions1, &firstPassPartitions1] {
        partitions1 = mergeAndCreatePartitionsVec<std::remove_cv_t<T1>>(
            firstPassPartitions1, returnBuffer1.get(), returnIndexes1.get());
        synchroniser.taskComplete();
      });
      threadPool.enqueue([this, &partitions2, &firstPassPartitions2] {
        partitions2 = mergeAndCreatePartitionsVec<std::remove_cv_t<T2>>(
            firstPassPartitions2, returnBuffer2.get(), returnIndexes2.get());
        synchroniser.taskComplete();
      });
      synchroniser.waitUntilComplete(2);
    } else {
      threadPool.enqueue([this, &partitions1, &firstPassPartitions1] {
        partitions1 = mergeKeysAndCreatePartitionsVec<std::remove_cv_t<T1>>(firstPassPartitions1,
                                                                            returnBuffer1.get());
        synchroniser.taskComplete();
      });
      threadPool.enqueue([this, &partitions2, &firstPassPartitions2] {
        partitions2 = mergeKeysAndCreatePartitionsVec<std::remove_cv_t<T2>>(firstPassPartitions2,
                                                                            returnBuffer2.get());
        synchroniser.taskComplete();
      });
      threadPool.enqueue([this, &firstPassPartitions1] {
        mergeIndexes<std::remove_cv_t<T1>>(firstPassPartitions1, returnIndexes1.get());
        synchroniser.taskComplete();
      });
      threadPool.enqueue([this, &firstPassPartitions2] {
        mergeIndexes<std::remove_cv_t<T2>>(firstPassPartitions2, returnIndexes2.get());
        synchroniser.taskComplete();
      });
      synchroniser.waitUntilComplete(4);
    }

#ifdef DEBUG
    std::cout << "Merged results from table1 first pass: " << std::endl;
    std::cout << "Partitions: " << std::endl;
    printArray<int>(partitions1.data(), static_cast<int>(partitions1.size()));
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T1>>(returnBuffer1.get(), nInput1);
    std::cout << "Indexes: " << std::endl;
    printArray<int32_t>(returnIndexes1.get(), nInput1);

    std::cout << "Merged results from table2 first pass: " << std::endl;
    std::cout << "Partitions: " << std::endl;
    printArray<int>(partitions2.data(), static_cast<int>(partitions2.size()));
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T2>>(returnBuffer2.get(), nInput2);
    std::cout << "Indexes: " << std::endl;
    printArray<int32_t>(returnIndexes2.get(), nInput2);
#endif

    int prevPartitionEnd1 = 0;
    int prevPartitionEnd2 = 0;
    if(msbToPartitionInput == 0) { // No ability to partition further, so return early
      outputPartitions1.reserve(partitions1.size());
      outputPartitions2.reserve(partitions1.size());
      for(size_t j = 0; j < partitions1.size(); ++j) {
        if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
        prevPartitionEnd1 = partitions1[j];
        prevPartitionEnd2 = partitions2[j];
      }
      return;
    }

    std::vector<std::unique_ptr<TwoPartitionedArraysPartitionsOnly>> furtherPartitioningResults(
        partitions1.size());
    int partitionsComplete = 0;
    int furtherPartitioningTasks = 0;

    for(size_t index = 0; index < partitions1.size(); ++index) {
      if(partitions1[index] != prevPartitionEnd1 && partitions2[index] != prevPartitionEnd2) {
        if((partitions1[index] - prevPartitionEnd1) > maxElementsPerPartition) {
          threadPool.enqueue([this, n1 = partitions1[index] - prevPartitionEnd1,
                              keys1 = returnBuffer1.get() + prevPartitionEnd1,
                              indexes1 = returnIndexes1.get() + prevPartitionEnd1,
                              prevPartitionEnd1, n2 = partitions2[index] - prevPartitionEnd2,
                              keys2 = returnBuffer2.get() + prevPartitionEnd2,
                              indexes2 = returnIndexes2.get() + prevPartitionEnd2,
                              prevPartitionEnd2, &furtherPartitioningResults, index] {
            auto op = getPartitionAdaptiveRawArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>>();
            auto partitionsPtr =
                std::make_unique<TwoPartitionedArraysPartitionsOnly>(op.processInput(
                    n1, keys1, indexes1, prevPartitionEnd1, n2, keys2, indexes2, prevPartitionEnd2,
                    minimumRadixBits, globalRadixBits, msbToPartitionInput, maxElementsPerPartition,
                    &totalOutputPartitions));
            {
              std::lock_guard<std::mutex> lock(resultsMutex);
              furtherPartitioningResults[index] = std::move(partitionsPtr);
            }
            synchroniser.taskComplete();
          });
          ++furtherPartitioningTasks;
        } else {
          ++partitionsComplete;
        }
      }
      prevPartitionEnd1 = partitions1[index];
      prevPartitionEnd2 = partitions2[index];
    }

    totalOutputPartitions.fetch_add(partitionsComplete);

    synchroniser.waitUntilComplete(furtherPartitioningTasks);

    outputPartitions1.reserve(totalOutputPartitions);
    outputPartitions2.reserve(totalOutputPartitions);

    prevPartitionEnd1 = 0;
    prevPartitionEnd2 = 0;
    for(size_t j = 0; j < partitions1.size(); ++j) {
      if(partitions1[j] != prevPartitionEnd1 && partitions2[j] != prevPartitionEnd2) {
        if((partitions1[j] - prevPartitionEnd1) > maxElementsPerPartition) {
          auto tableOnePartitionPositions =
              furtherPartitioningResults[j]->tableOnePartitionPositions.get();
          auto tableTwoPartitionPositions =
              furtherPartitioningResults[j]->tableTwoPartitionPositions.get();
          std::move(tableOnePartitionPositions->begin(), tableOnePartitionPositions->end(),
                    std::back_inserter(outputPartitions1));
          std::move(tableTwoPartitionPositions->begin(), tableTwoPartitionPositions->end(),
                    std::back_inserter(outputPartitions2));
        } else {
          outputPartitions1.emplace_back(prevPartitionEnd1, partitions1[j] - prevPartitionEnd1);
          outputPartitions2.emplace_back(prevPartitionEnd2, partitions2[j] - prevPartitionEnd2);
        }
      }
      prevPartitionEnd1 = partitions1[j];
      prevPartitionEnd2 = partitions2[j];
    }
  }

  template <typename T>
  std::vector<int>
  mergeAndCreatePartitionsVec(const std::vector<PartitionedArrayAlt<T>>& partitionedArrays,
                              T* keyBuffer, int32_t* indexesBuffer) {
    std::vector<int> mergedPartitions;
    int numPartitions = partitionedArrays[0].partitionPositions->size() - 1;
    mergedPartitions.reserve(numPartitions);

    int totalElements = 0;
    for(int partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
      for(size_t threadNum = 0; threadNum < partitionedArrays.size(); ++threadNum) {
#ifdef DEBUG
        assert(static_cast<int>(partitionedArrays[threadNum].partitionPositions->size()) - 1 ==
               numPartitions);
#endif
        int threadPartitionElements =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum + 1) -
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
        if(threadPartitionElements == 0) {
          continue;
        }
        int threadPartitionElementsStart =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
#ifdef DEBUG
        std::cout << "Partition number " << partitionNum << " , thread number " << threadNum
                  << std::endl;
        std::cout << "Keys: " << std::endl;
        printArray<T>(&(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
                      threadPartitionElements);
        std::cout << "Indexes: " << std::endl;
        printArray<int32_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
                            threadPartitionElements);
#endif
        memcpy(keyBuffer + totalElements,
               &(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(T));
        memcpy(indexesBuffer + totalElements,
               &(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(int32_t));

        totalElements += threadPartitionElements;
      }
      mergedPartitions.push_back(totalElements);
    }

    return mergedPartitions;
  }

  template <typename T>
  std::vector<int>
  mergeKeysAndCreatePartitionsVec(const std::vector<PartitionedArrayAlt<T>>& partitionedArrays,
                                  T* keyBuffer) {
    std::vector<int> mergedPartitions;
    int numPartitions = partitionedArrays[0].partitionPositions->size() - 1;
    mergedPartitions.reserve(numPartitions);

    int totalElements = 0;
    for(int partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
      for(size_t threadNum = 0; threadNum < partitionedArrays.size(); ++threadNum) {
#ifdef DEBUG
        assert(static_cast<int>(partitionedArrays[threadNum].partitionPositions->size()) - 1 ==
               numPartitions);
#endif
        int threadPartitionElements =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum + 1) -
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
        if(threadPartitionElements == 0) {
          continue;
        }
        int threadPartitionElementsStart =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
#ifdef DEBUG
        std::cout << "Partition number " << partitionNum << " , thread number " << threadNum
                  << std::endl;
        std::cout << "Keys: " << std::endl;
        printArray<T>(&(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
                      threadPartitionElements);
        std::cout << "Indexes: " << std::endl;
        printArray<int32_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
                            threadPartitionElements);
#endif
        memcpy(keyBuffer + totalElements,
               &(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(T));

        totalElements += threadPartitionElements;
      }
      mergedPartitions.push_back(totalElements);
    }

    return mergedPartitions;
  }

  template <typename T>
  void mergeIndexes(const std::vector<PartitionedArrayAlt<T>>& partitionedArrays,
                    int32_t* indexesBuffer) {
    int numPartitions = partitionedArrays[0].partitionPositions->size() - 1;

    int totalElements = 0;
    for(int partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
      for(size_t threadNum = 0; threadNum < partitionedArrays.size(); ++threadNum) {
#ifdef DEBUG
        assert(static_cast<int>(partitionedArrays[threadNum].partitionPositions->size()) - 1 ==
               numPartitions);
#endif
        int threadPartitionElements =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum + 1) -
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
        if(threadPartitionElements == 0) {
          continue;
        }
        int threadPartitionElementsStart =
            partitionedArrays[threadNum].partitionPositions->at(partitionNum);
#ifdef DEBUG
        std::cout << "Partition number " << partitionNum << " , thread number " << threadNum
                  << std::endl;
        std::cout << "Keys: " << std::endl;
        printArray<T>(&(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
                      threadPartitionElements);
        std::cout << "Indexes: " << std::endl;
        printArray<int32_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
                            threadPartitionElements);
#endif
        memcpy(indexesBuffer + totalElements,
               &(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(int32_t));

        totalElements += threadPartitionElements;
      }
    }
  }

  template <typename T>
  inline std::vector<int> createCumulativeBatchSizes(const ExpressionSpanArguments& spans) {
    std::vector<int> cumulativeBatchSizes(1, 0);
    cumulativeBatchSizes.reserve(spans.size() + 1);
    for(const auto& batch : spans) {
      cumulativeBatchSizes.push_back(cumulativeBatchSizes.back() + get<Span<T>>(batch).size());
    }
    return cumulativeBatchSizes;
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
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::shared_ptr<int32_t[]> returnIndexes1;
  vectorOfPairs<int, int> outputPartitions1;

  int nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::shared_ptr<int32_t[]> returnIndexes2;
  vectorOfPairs<int, int> outputPartitions2;

  int dop;
  int minimumRadixBits;
  int startingRadixBits;
  std::atomic<int> globalRadixBits;
  int msbToPartitionInput{};
  int maxElementsPerPartition;
  std::atomic<int> threadsStillRunning;
  ThreadPool& threadPool;
  Synchroniser& synchroniser;
  std::atomic<int> totalOutputPartitions;

  std::mutex resultsMutex;
};

/********************************** UTILITY FUNCTIONS *********************************/

#ifdef INCLUDE_MIN_PARTITION_SIZE

template <typename T1, typename T2>
PartitionedJoinArguments
createPartitionsWithMinimumSize(const TwoPartitionedArrays<T1, T2>& partitionedTables) {

  auto keys1 = partitionedTables.partitionedArrayOne.partitionedKeys.get();
  auto indexes1 = partitionedTables.partitionedArrayOne.indexes.get();
  auto partitions1 = *(partitionedTables.partitionedArrayOne.partitionPositions.get());

  auto keys2 = partitionedTables.partitionedArrayTwo.partitionedKeys.get();
  auto indexes2 = partitionedTables.partitionedArrayTwo.indexes.get();
  auto partitions2 = *(partitionedTables.partitionedArrayTwo.partitionPositions.get());

  std::vector<ExpressionSpanArguments> partitionsOfKeySpans1, partitionsOfIndexSpans1,
      partitionsOfKeySpans2, partitionsOfIndexSpans2;
  partitionsOfKeySpans1.reserve(partitions1.size());
  partitionsOfIndexSpans1.reserve(partitions1.size());
  partitionsOfKeySpans2.reserve(partitions1.size());
  partitionsOfIndexSpans2.reserve(partitions1.size());

  size_t partitionNum = 0;
  while(partitionNum < partitions1.size()) {
    int partitionStart1 = partitions1[partitionNum].first;
    int partitionStart2 = partitions2[partitionNum].first;
    int partitionSize1 = partitions1[partitionNum].second;
    int partitionSize2 = partitions2[partitionNum].second;
    ++partitionNum;

    while(partitionSize1 < MIN_PARTITION_SIZE && partitionNum < partitions1.size()) {
      partitionSize1 =
          (partitions1[partitionNum].first - partitionStart1) + partitions1[partitionNum].second;
      partitionSize2 =
          (partitions2[partitionNum].first - partitionStart2) + partitions2[partitionNum].second;
      ++partitionNum;
    }

    partitionsOfKeySpans1.emplace_back(
        Span<T1>(keys1 + partitionStart1, partitionSize1,
                 [ptr = partitionedTables.partitionedArrayOne.partitionedKeys]() {}));
    partitionsOfIndexSpans1.emplace_back(
        Span<int32_t>(indexes1 + partitionStart1, partitionSize1,
                      [ptr = partitionedTables.partitionedArrayOne.indexes]() {}));

    partitionsOfKeySpans2.emplace_back(
        Span<T2>(keys2 + partitionStart2, partitionSize2,
                 [ptr = partitionedTables.partitionedArrayTwo.partitionedKeys]() {}));
    partitionsOfIndexSpans2.emplace_back(
        Span<int32_t>(indexes2 + partitionStart2, partitionSize2,
                      [ptr = partitionedTables.partitionedArrayTwo.indexes]() {}));
  }

  return {std::move(partitionsOfKeySpans1), std::move(partitionsOfIndexSpans1),
          std::move(partitionsOfKeySpans2), std::move(partitionsOfIndexSpans2)};
}

#else

#ifdef CREATE_SPANS_ALIGNED_TO_BATCHES
template <typename T>
std::pair<std::vector<ExpressionSpanArguments>, std::vector<ExpressionSpanArguments>>
createPartitionsOfSpansAlignedToTableBatches(
    const PartitionedArray<std::remove_cv_t<T>>& partitionedArray,
    const ExpressionSpanArguments& tableKeys) {
#else
template <typename T>
std::pair<std::vector<ExpressionSpanArguments>, std::vector<ExpressionSpanArguments>>
createPartitionsOfSpansAlignedToTableBatches(
    const PartitionedArray<std::remove_cv_t<T>>& partitionedArray,
    const ExpressionSpanArguments& /*unused*/) {
#endif

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

#ifdef CREATE_SPANS_ALIGNED_TO_BATCHES
  std::vector<int> cumulativeBatchSizes(1, 0);
  cumulativeBatchSizes.reserve(tableKeys.size() + 1);
  for(const auto& batch : tableKeys) {
    cumulativeBatchSizes.push_back(cumulativeBatchSizes.back() + get<Span<T>>(batch).size());
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
        outputKeySpans.emplace_back(Span<std::remove_cv_t<T>>());
        outputIndexSpans.emplace_back(Span<int32_t>());
      } else {
        outputKeySpans.emplace_back(
            Span<std::remove_cv_t<T>>(keys + prevSpanEndIndex, index - prevSpanEndIndex,
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
#else
  for(auto& partition : partitions) {
    ExpressionSpanArguments outputKeySpans, outputIndexSpans;
    outputKeySpans.emplace_back(Span<std::remove_cv_t<T>>(
        keys + partition.first, partition.second, [ptr = partitionedArray.partitionedKeys]() {}));
    outputIndexSpans.emplace_back(Span<int32_t>(indexes + partition.first, partition.second,
                                                [ptr = partitionedArray.indexes]() {}));
    partitionsOfKeySpans.push_back(std::move(outputKeySpans));
    partitionsOfIndexSpans.push_back(std::move(outputIndexSpans));
  }
#endif

  return std::make_pair(std::move(partitionsOfKeySpans), std::move(partitionsOfIndexSpans));
}

#endif

/*********************************** ENTRY FUNCTION ***********************************/

template <typename T1, typename T2>
PartitionedJoinArguments partitionJoinExpr(PartitionOperators partitionImplementation,
                                           const ExpressionSpanArguments& tableOneKeys,
                                           const ExpressionSpanArguments& tableTwoKeys, int dop) {
  static_assert(std::is_integral<T1>::value, "PartitionOperators column must be an integer type");
  static_assert(std::is_integral<T2>::value, "PartitionOperators column must be an integer type");

  TwoPartitionedArrays<std::remove_cv_t<T1>, std::remove_cv_t<T2>> partitionedTables =
      [partitionImplementation, &tableOneKeys, &tableTwoKeys, dop]() {
#ifdef USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1
        if(partitionImplementation == PartitionOperators::RadixBitsAdaptiveParallel && dop == 1) {
          auto partitionOperator = PartitionAdaptive<T1, T2>(tableOneKeys, tableTwoKeys);
          return partitionOperator.processInput();
        }
#endif
        if(partitionImplementation == PartitionOperators::RadixBitsAdaptiveParallel) {
          assert(adaptive::config::nonVectorizedDOP >= dop); // Will have a deadlock otherwise
          auto partitionOperator =
              PartitionAdaptiveParallel<T1, T2>(tableOneKeys, tableTwoKeys, dop);
          return partitionOperator.processInput();
        }
        assert(dop == 1);
        if(partitionImplementation == PartitionOperators::RadixBitsFixedMin) {
          std::string name = "Partition_minRadixBits";
          auto radixBitsMin =
              static_cast<int>(MachineConstants::getInstance().getMachineConstant(name));
          auto partitionOperator = Partition<T1, T2>(tableOneKeys, tableTwoKeys, radixBitsMin);
          return partitionOperator.processInput();
        } else if(partitionImplementation == PartitionOperators::RadixBitsFixedMax) {
          std::string name = "Partition_startRadixBits";
          auto radixBitsMax =
              static_cast<int>(MachineConstants::getInstance().getMachineConstant(name));
          auto partitionOperator = Partition<T1, T2>(tableOneKeys, tableTwoKeys, radixBitsMax);
          return partitionOperator.processInput();
        } else if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
          auto partitionOperator = PartitionAdaptive<T1, T2>(tableOneKeys, tableTwoKeys);
          return partitionOperator.processInput();
        } else {
          throw std::runtime_error("Invalid selection of 'Partition' implementation!");
        }
      }();

#ifdef INCLUDE_MIN_PARTITION_SIZE

  return createPartitionsWithMinimumSize(partitionedTables);

#else

  std::vector<ExpressionSpanArguments> tableOnePartitionsOfKeySpans, tableOnePartitionsOfIndexSpans,
      tableTwoPartitionsOfKeySpans, tableTwoPartitionsOfIndexSpans;
  if(dop == 1) {
    auto [tableOnePartitionsOfKeySpansTmp, tableOnePartitionsOfIndexSpansTmp] =
        createPartitionsOfSpansAlignedToTableBatches<T1>(partitionedTables.partitionedArrayOne,
                                                         tableOneKeys);
    tableOnePartitionsOfKeySpans = std::move(tableOnePartitionsOfKeySpansTmp);
    tableOnePartitionsOfIndexSpans = std::move(tableOnePartitionsOfIndexSpansTmp);
    auto [tableTwoPartitionsOfKeySpansTmp, tableTwoPartitionsOfIndexSpansTmp] =
        createPartitionsOfSpansAlignedToTableBatches<T2>(partitionedTables.partitionedArrayTwo,
                                                         tableTwoKeys);
    tableTwoPartitionsOfKeySpans = std::move(tableTwoPartitionsOfKeySpansTmp);
    tableTwoPartitionsOfIndexSpans = std::move(tableTwoPartitionsOfIndexSpansTmp);
  } else {
    ThreadPool::getInstance().enqueue([&tableOneKeys, &partitionedTables,
                                       &tableOnePartitionsOfKeySpans,
                                       &tableOnePartitionsOfIndexSpans, &synchroniser] {
      auto [tableOnePartitionsOfKeySpansTmp, tableOnePartitionsOfIndexSpansTmp] =
          createPartitionsOfSpansAlignedToTableBatches<T1>(partitionedTables.partitionedArrayOne,
                                                           tableOneKeys);
      tableOnePartitionsOfKeySpans = std::move(tableOnePartitionsOfKeySpansTmp);
      tableOnePartitionsOfIndexSpans = std::move(tableOnePartitionsOfIndexSpansTmp);
      synchroniser.taskComplete();
    });
    ThreadPool::getInstance().enqueue([&tableTwoKeys, &partitionedTables,
                                       &tableTwoPartitionsOfKeySpans,
                                       &tableTwoPartitionsOfIndexSpans, &synchroniser] {
      auto [tableTwoPartitionsOfKeySpansTmp, tableTwoPartitionsOfIndexSpansTmp] =
          createPartitionsOfSpansAlignedToTableBatches<T1>(partitionedTables.partitionedArrayTwo,
                                                           tableTwoKeys);
      tableTwoPartitionsOfKeySpans = std::move(tableTwoPartitionsOfKeySpansTmp);
      tableTwoPartitionsOfIndexSpans = std::move(tableTwoPartitionsOfIndexSpansTmp);
      synchroniser.taskComplete();
    });
    synchroniser.waitUntilComplete(2);
  }

  return {std::move(tableOnePartitionsOfKeySpans), std::move(tableOnePartitionsOfIndexSpans),
          std::move(tableTwoPartitionsOfKeySpans), std::move(tableTwoPartitionsOfIndexSpans)};

#endif
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITIONIMPLEMENTATION_HPP
