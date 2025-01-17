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

#include "HazardAdaptiveEngine.hpp"
#include "constants/machineConstants.hpp"
#include "utilities/dataStructures.hpp"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/systemInformation.hpp"
#include "utilities/utilities.hpp"

#define USE_ADAPTIVE_OVER_ADAPTIVE_PARALLEL_FOR_DOP_1
// #define CREATE_SPANS_ALIGNED_TO_BATCHES
#define ADAPTIVITY_OUTPUT
// #define DEBUG
// #define CHANGE_PARTITION_TO_SORT_FOR_TESTING

#ifndef CREATE_SPANS_ALIGNED_TO_BATCHES
#define INCLUDE_MIN_PARTITION_SIZE // Toggle, requires not creating spans aligned to batches
#endif

// NOLINTBEGIN(hicpp-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays,
// readability-function-cognitive-complexity)

namespace adaptive {

namespace partitionConfig {
constexpr int TUPLES_PER_HAZARD_CHECK = 10'000;
constexpr float TUPLES_PER_STLB_STORE_MISS = 10.0;
constexpr uint32_t SPAN_OFFSET_BITS = 32;
} // namespace partitionConfig

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
      : sTlbStoreMisses(sTlbStoreMisses_),
        tuplesPerTlbStoreMiss(partitionConfig::TUPLES_PER_STLB_STORE_MISS) {}
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
  std::shared_ptr<int64_t[]> indexes;
  std::unique_ptr<vectorOfPairs<int, int>> partitionPositions; // {start, size}
};

template <typename T1, typename T2> struct TwoPartitionedArrays {
  PartitionedArray<T1> partitionedArrayOne;
  PartitionedArray<T2> partitionedArrayTwo;
};

template <typename T1, typename T2> class Partition {
public:
  Partition(const ExpressionSpanArguments& keySpans1_, const ExpressionSpanArguments& keySpans2_,
            uint32_t radixBitsInput = 0)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_) {
    std::string minName = "Partition_minRadixBits";
    auto radixBitsMin =
        static_cast<uint32_t>(MachineConstants::getInstance().getMachineConstant(minName));
    radixBitsOperator = std::max(radixBitsInput, radixBitsMin);

    nInput1 = getTotalLength<T1>(keySpans1);
    nInput2 = getTotalLength<T2>(keySpans2);

#ifdef CHANGE_PARTITION_TO_SORT_FOR_TESTING
    maxElementsPerPartition = 1; // NOLINT
#else
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / sizeof(T1); // NOLINT
#endif

    buckets1 = std::vector<int>(1 + (1U << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1); // NOLINT
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed // NOLINT

    returnIndexes1 = std::make_shared_for_overwrite<int64_t[]>(nInput1); // NOLINT
    tmpIndexes1 = std::make_unique_for_overwrite<int64_t[]>(nInput1);    // NOLINT
    uint32_t msbToPartitionInput1 = getMsb<T1>(keySpans1, tmpIndexes1.get());

    buckets2 = std::vector<int>(1 + (1U << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2); // NOLINT
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed // NOLINT

    returnIndexes2 = std::make_shared_for_overwrite<int64_t[]>(nInput2); // NOLINT
    tmpIndexes2 = std::make_unique_for_overwrite<int64_t[]>(nInput2);    // NOLINT
    uint32_t msbToPartitionInput2 = getMsb<T2>(keySpans2, tmpIndexes2.get());

    msbToPartitionInput = std::max(msbToPartitionInput1, msbToPartitionInput2);
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
  inline void performPartition() { // NOLINT
    uint32_t msbToPartition = msbToPartitionInput;
    auto buffer1 = returnBuffer1.get();
    auto* indexes1 = tmpIndexes1.get();
    auto* indexesBuffer1 = returnIndexes1.get();
    auto* buffer2 = returnBuffer2.get();
    auto* indexes2 = tmpIndexes2.get();
    auto* indexesBuffer2 = returnIndexes2.get();

    uint32_t radixBits = std::min(msbToPartition, radixBitsOperator);
    uint32_t shifts = msbToPartition - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t mask = numBuckets - 1;

    // Complete partitioning for array 1
    uint32_t i = 0;
    for(const auto& span : keySpans1) {
      for(auto& key : std::get<Span<T1>>(span)) {
        buckets1[1 + ((key >> shifts) & mask)]++;
      }
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);

    int offset = 0;
    for(const auto& untypedSpan : keySpans1) {
      auto& span = std::get<Span<T1>>(untypedSpan);
      for(i = 0; i < span.size(); i++) {
        auto index = buckets1[(span[i] >> shifts) & mask]++;
        buffer1[index] = span[i];
        indexesBuffer1[index] = indexes1[offset + i];
      }
      offset += span.size();
    }
    std::fill(buckets1.begin(), buckets1.end(), 0);

    // Complete partitioning for array 2
    for(const auto& span : keySpans2) {
      for(auto& key : std::get<Span<T2>>(span)) {
        buckets2[1 + ((key >> shifts) & mask)]++;
      }
    }

    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);

    offset = 0;
    for(const auto& untypedSpan : keySpans2) {
      auto& span = std::get<Span<T2>>(untypedSpan);
      for(i = 0; i < span.size(); i++) {
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

  inline void performPartitionAux(uint32_t n1, std::remove_cv_t<T1>* keys1, // NOLINT
                                  std::remove_cv_t<T1>* buffer1, int64_t* indexes1,
                                  int64_t* indexesBuffer1, int offset1, uint32_t n2,
                                  std::remove_cv_t<T2>* keys2, std::remove_cv_t<T2>* buffer2,
                                  int64_t* indexes2, int64_t* indexesBuffer2, int offset2,
                                  uint32_t msbToPartition, uint32_t radixBits, bool copyRequired) {
    radixBits = std::min(msbToPartition, radixBits);
    uint32_t shifts = msbToPartition - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t mask = numBuckets - 1;

    // Complete partitioning for array 1
    uint32_t i = 0;
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
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int64_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int64_t));
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
                        (partitions1[j] - prevPartitionEnd1) * sizeof(int64_t));
            std::memcpy(keys2 + prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(T2));
            std::memcpy(indexes2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(int64_t));
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

  template <typename U> inline int getTotalLength(const ExpressionSpanArguments& keySpans) {
    uint32_t n = 0;
    for(const auto& untypedSpan : keySpans) {
      const auto& span = std::get<Span<U>>(untypedSpan);
      n += span.size();
    }
    return n;
  }

  template <typename U>
  inline uint32_t getMsb(const ExpressionSpanArguments& keySpans, int64_t* indexes) {
    auto largest = std::numeric_limits<U>::min();
    uint64_t spanNumber = 0;
    uint32_t indexNumber = 0;
    for(const auto& untypedSpan : keySpans) {
      const auto& span = std::get<Span<U>>(untypedSpan);
      uint32_t spanOffset = 0;
      for(const auto& key : span) {
        largest = std::max(largest, key);
        indexes[indexNumber++] =
            static_cast<int64_t>((spanNumber << partitionConfig::SPAN_OFFSET_BITS) | spanOffset++);
      }
      spanNumber++;
    }

    uint32_t msbToPartition = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartition++;
    }
    return msbToPartition;
  }

  uint32_t nInput1;
  const ExpressionSpanArguments& keySpans1;
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int64_t[]> returnIndexes1;
  std::unique_ptr<int64_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  uint32_t nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
  std::shared_ptr<int64_t[]> returnIndexes2;
  std::unique_ptr<int64_t[]> tmpIndexes2;
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;

  uint32_t radixBitsOperator;
  uint32_t msbToPartitionInput;
  int maxElementsPerPartition;
};

/****************************** SINGLE-THREADED ******************************/

// This class acts as the Dispatcher
template <typename T1, typename T2> class PartitionAdaptive {
public:
  PartitionAdaptive(const ExpressionSpanArguments& keySpans1_,
                    const ExpressionSpanArguments& keySpans2_)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_),
        eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(partitionConfig::TUPLES_PER_HAZARD_CHECK) {
    std::string startName = "Partition_startRadixBits";
    radixBitsOperator =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(startName));

    std::string minName = "Partition_minRadixBits";
    minimumRadixBits =
        static_cast<int>(MachineConstants::getInstance().getMachineConstant(minName));

    nInput1 = getTotalLength<T1>(keySpans1);
    nInput2 = getTotalLength<T2>(keySpans2);

#ifdef CHANGE_PARTITION_TO_SORT_FOR_TESTING
    maxElementsPerPartition = 1; // NOLINT
#else
    maxElementsPerPartition = static_cast<double>(l2cacheSize()) / sizeof(T1); // NOLINT
#endif

    buckets1 = std::vector<int>(1 + (1U << radixBitsOperator), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1); // NOLINT
    tmpBuffer1 = nullptr; // Lazily allocate buffer when needed // NOLINT

    returnIndexes1 = std::make_shared_for_overwrite<int64_t[]>(nInput1); // NOLINT
    tmpIndexes1 = std::make_unique_for_overwrite<int64_t[]>(nInput1);    // NOLINT
    uint32_t msbToPartitionInput1 = getMsb<T1>(keySpans1, tmpIndexes1.get());

    buckets2 = std::vector<int>(1 + (1U << radixBitsOperator), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2); // NOLINT
    tmpBuffer2 = nullptr; // Lazily allocate buffer when needed // NOLINT

    returnIndexes2 = std::make_shared_for_overwrite<int64_t[]>(nInput2); // NOLINT
    tmpIndexes2 = std::make_unique_for_overwrite<int64_t[]>(nInput2);    // NOLINT
    uint32_t msbToPartitionInput2 = getMsb<T2>(keySpans2, tmpIndexes2.get());

    msbToPartitionInput = std::max(msbToPartitionInput1, msbToPartitionInput2);
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
  inline void performPartition() { // NOLINT
    uint32_t msbToPartition = msbToPartitionInput;
    auto buffer1 = returnBuffer1.get(); // Output Collector
    auto* indexes1 = tmpIndexes1.get();
    auto* indexesBuffer1 = returnIndexes1.get();
    auto* buffer2 = returnBuffer2.get(); // Output Collector
    auto* indexes2 = tmpIndexes2.get();
    auto* indexesBuffer2 = returnIndexes2.get();

    uint32_t radixBits = std::min(msbToPartition, radixBitsOperator);
    uint32_t shifts = msbToPartition - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t startingNumBuckets = numBuckets;
    uint32_t mask = numBuckets - 1;

    // Complete histogram for array 1
    uint32_t i = 0;
    for(const auto& span : keySpans1) {
      for(auto& key : std::get<Span<T1>>(span)) {
        buckets1[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets1[i] += buckets1[i - 1];
    }
    std::vector<int> partitions1(buckets1.data() + 1, buckets1.data() + numBuckets + 1);

    // Complete histogram for array 2
    for(const auto& span : keySpans2) {
      for(auto& key : std::get<Span<T2>>(span)) {
        buckets2[1 + ((key >> shifts) & mask)]++;
      }
    }
    for(i = 2; i <= numBuckets; i++) {
      buckets2[i] += buckets2[i - 1];
    }
    std::vector<int> partitions2(buckets2.data() + 1, buckets2.data() + numBuckets + 1);

    bool keysComplete1 = false;
    bool keysComplete2 = false;
    auto outerIterator1 = keySpans1.begin();
    auto innerIterator1 = (std::get<Span<T1>>(*outerIterator1)).begin();
    auto outerIterator2 = keySpans2.begin();
    auto innerIterator2 = (std::get<Span<T2>>(*outerIterator2)).begin();
    int offset1 = 0;
    int offset2 = 0;

    // This loop combined with the outer and inner iterators acts as the Input Interleaver
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

  inline void performPartitionAux(uint32_t n1, std::remove_cv_t<T1>* keys1,
                                  std::remove_cv_t<T1>* buffer1, int64_t* indexes1,
                                  int64_t* indexesBuffer1, int offset1, uint32_t n2,
                                  std::remove_cv_t<T2>* keys2, std::remove_cv_t<T2>* buffer2,
                                  int64_t* indexes2, int64_t* indexesBuffer2, int offset2,
                                  uint32_t msbToPartition, uint32_t radixBits, bool copyRequired) {
    radixBits = std::min(msbToPartition, radixBits);
    uint32_t shifts = msbToPartition - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t startingNumBuckets = numBuckets;
    uint32_t mask = numBuckets - 1;

    // Complete histogram for array 1
    uint32_t i, microBatchStart, microBatchSize; // NOLINT
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

    // This loop acts as the Input Interleaver
    uint32_t i1 = 0, i2 = 0; // NOLINT
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
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int64_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int64_t));
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
                        (partitions1[j] - prevPartitionEnd1) * sizeof(int64_t));
            std::memcpy(keys2 + prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(T2));
            std::memcpy(indexes2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(int64_t));
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
      std::remove_cv_t<U1>* buffer, const int64_t* indexes, int64_t* indexesBuffer,
      std::vector<int>& buckets, std::vector<int>& partitions, int& offset, bool& keysComplete,
      uint32_t& shifts, uint32_t& mask, uint32_t& radixBits, uint32_t& numBuckets,
      OuterIt_2& outerIterator_2, InnerIt_2& innerIterator_2,
      const ExpressionSpanArguments& keySpans_2, std::remove_cv_t<U2>* buffer_2,
      const int64_t* indexes_2, int64_t* indexesBuffer_2, std::vector<int>& buckets_2,
      std::vector<int>& partitions_2, int& offset_2, bool& keysComplete_2) {
    uint32_t processed = 0;
    uint32_t microBatchChunkSize; // NOLINT
    eventSet.readCounters();
    while(processed < tuplesPerHazardCheck) {
      microBatchChunkSize = std::min(
          tuplesPerHazardCheck - processed,
          static_cast<uint32_t>((std::get<Span<U1>>(*outerIterator)).end() - innerIterator));
      for(auto i = 0U; i < microBatchChunkSize; ++i) { // Run chunk
        auto index = buckets[((*innerIterator) >> shifts) & mask]++;
        buffer[index] = *(innerIterator++);
        indexesBuffer[index] = indexes[offset + processed++];
      }
      if(innerIterator == (std::get<Span<U1>>(*outerIterator)).end()) {
        if(++outerIterator == keySpans.end()) {
          keysComplete = true;
          break;
        } else { // NOLINT
          innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
        }
      }
    }
    eventSet.readCountersAndUpdateDiff();
    offset += processed;

    // This calls the Output Transformer
    if(processed == tuplesPerHazardCheck && radixBits > minimumRadixBits &&
       monitor.robustnessIncreaseRequired(tuplesPerHazardCheck)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1U;
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
          } else { // NOLINT
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
          } else { // NOLINT
            innerIterator_2 = (std::get<Span<U2>>(*outerIterator_2)).begin();
          }
        }
      }

      keysComplete = true;
      keysComplete_2 = true;
    }
  }

  template <typename U1, typename U2>
  inline void
  processMicroBatch(uint32_t microBatchStart, uint32_t microBatchSize, uint32_t& i, uint32_t n,
                    const U1* keys, std::remove_cv_t<U1>* buffer, const int64_t* indexes,
                    int64_t* indexesBuffer, std::vector<int>& buckets, std::vector<int>& partitions,
                    uint32_t& shifts, uint32_t& mask, uint32_t& radixBits, uint32_t& numBuckets,
                    uint32_t& i_2, uint32_t n_2, const U2* keys_2, std::remove_cv_t<U2>* buffer_2,
                    const int64_t* indexes_2, int64_t* indexesBuffer_2, std::vector<int>& buckets_2,
                    std::vector<int>& partitions_2) {
    eventSet.readCounters();

    for(; i < microBatchStart + microBatchSize; i++) { // Run chunk
      auto index = buckets[(keys[i] >> shifts) & mask]++;
      buffer[index] = keys[i];
      indexesBuffer[index] = indexes[i];
    }

    eventSet.readCountersAndUpdateDiff();

    // This calls the Output Transformer
    if(microBatchSize == tuplesPerHazardCheck &&
       monitor.robustnessIncreaseRequired(microBatchSize)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1U;
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

  // This function represents the Output Transformer
  template <typename U>
  inline void mergePartitions(U* buffer, int64_t* indexesBuffer, std::vector<int>& partitions,
                              std::vector<int>& buckets, uint32_t numBuckets, bool valuesInBuffer) {
    if(valuesInBuffer) {                         // Skip if no elements have been scattered yet
      for(uint32_t j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1U];
        auto srcIndex = partitions[j << 1U];
        auto numElements = buckets[(j << 1U) + 1] - srcIndex;
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int64_t));
      }
    }

    for(uint32_t j = 0; j < numBuckets; ++j) { // Merge histogram values
      buckets[j] = buckets[j << 1U] + (buckets[(j << 1U) + 1] - partitions[j << 1U]);
    }

    for(uint32_t j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j - 1] = partitions[(j << 1U) - 1];
    }
    partitions.resize(numBuckets);
  }

  template <typename U> inline int getTotalLength(const ExpressionSpanArguments& keySpans) {
    uint32_t n = 0;
    for(const auto& untypedSpan : keySpans) {
      const auto& span = std::get<Span<U>>(untypedSpan);
      n += span.size();
    }
    return n;
  }

  template <typename U>
  inline uint32_t getMsb(const ExpressionSpanArguments& keySpans, int64_t* indexes) {
    auto largest = std::numeric_limits<U>::min();
    uint64_t spanNumber = 0;
    uint32_t indexNumber = 0;
    for(const auto& untypedSpan : keySpans) {
      const auto& span = std::get<Span<U>>(untypedSpan);
      uint32_t spanOffset = 0;
      for(const auto& key : span) {
        largest = std::max(largest, key);
        indexes[indexNumber++] =
            static_cast<int64_t>((spanNumber << partitionConfig::SPAN_OFFSET_BITS) | spanOffset++);
      }
      spanNumber++;
    }

    uint32_t msbToPartition = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartition++;
    }
    return msbToPartition;
  }

  uint32_t nInput1;
  const ExpressionSpanArguments& keySpans1;
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int64_t[]> returnIndexes1;
  std::unique_ptr<int64_t[]> tmpIndexes1;
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;

  uint32_t nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
  std::shared_ptr<int64_t[]> returnIndexes2;
  std::unique_ptr<int64_t[]> tmpIndexes2;
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;

  uint32_t minimumRadixBits;
  uint32_t radixBitsOperator;
  uint32_t msbToPartitionInput;
  int maxElementsPerPartition;

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  uint32_t tuplesPerHazardCheck;
};

/************************** SINGLE-THREADED FOR MULTI-THREADED **************************/

struct TwoPartitionedArraysPartitionsOnly {
  std::unique_ptr<vectorOfPairs<int, int>> tableOnePartitionPositions;
  std::unique_ptr<vectorOfPairs<int, int>> tableTwoPartitionPositions;
};

template <typename T> struct PartitionedArrayAlt {
  std::shared_ptr<T[]> partitionedKeys;
  std::shared_ptr<int64_t[]> indexes;
  std::unique_ptr<std::vector<int>> partitionPositions;
};

template <typename T1, typename T2> struct TwoPartitionedArraysAlt {
  PartitionedArrayAlt<T1> partitionedArrayOne;
  PartitionedArrayAlt<T2> partitionedArrayTwo;
};

// This class acts as the Dispatcher
template <typename T1, typename T2> class PartitionAdaptiveRawArrays {
public:
  PartitionAdaptiveRawArrays()
      : eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(partitionConfig::TUPLES_PER_HAZARD_CHECK) {

    size_t initialSize = 2 * l2cacheSize() / 4;
    keysTmpBuffer1.resize(initialSize);
    indexesTmpBuffer1.resize(initialSize);
    keysTmpBuffer2.resize(initialSize);
    indexesTmpBuffer2.resize(initialSize);
  }

  TwoPartitionedArraysPartitionsOnly
  processInput(uint32_t n1, T1* keys1, int64_t* indexes1, int overallOffset1_, uint32_t n2,
               T2* keys2, int64_t* indexes2, int overallOffset2_, uint32_t minimumRadixBits_,
               uint32_t radixBits, uint32_t msbToPartitionInput_, int maxElementsPerPartition_,
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

    buckets1.resize(1 + (1U << radixBitsOperator));
    std::fill(buckets1.begin(), buckets1.end(), 0);
    buckets2.resize(1 + (1U << radixBitsOperator));
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
  inline void performPartition(uint32_t n1, T1* keys1, T1* buffer1, int64_t* indexes1,
                               int64_t* indexesBuffer1, int offset1, uint32_t n2, T2* keys2,
                               T2* buffer2, int64_t* indexes2, int64_t* indexesBuffer2, int offset2,
                               uint32_t msbToPartition, uint32_t radixBits, bool copyRequired) {
    radixBits = std::min(msbToPartition, radixBits);
    uint32_t shifts = msbToPartition - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t startingNumBuckets = numBuckets;
    uint32_t mask = numBuckets - 1;

    // Complete histogram for array 1
    uint32_t i, microBatchStart, microBatchSize; // NOLINT
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

    // This loop acts as the Input Interleaver
    uint32_t i1 = 0, i2 = 0; // NOLINT
    if(radixBits > minimumRadixBits) {
      while(i2 < n2 || i1 < n1) {
        if(i1 < n1) {
          microBatchSize = std::min(tuplesPerHazardCheck, n1 - i1);
          microBatchStart = i1;

          processMicroBatch<T1, T2>(microBatchStart, microBatchSize, i1, n1, keys1, buffer1,
                                    indexes1, indexesBuffer1, buckets1, partitions1, shifts, mask,
                                    radixBits, numBuckets, i2, n2, keys2, buffer2, indexes2,
                                    indexesBuffer2, buckets2, partitions2);
        }
        if(i2 < n2) {
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
        std::memcpy(indexes1, indexesBuffer1, n1 * sizeof(int64_t));
        std::memcpy(keys2, buffer2, n2 * sizeof(T2));
        std::memcpy(indexes2, indexesBuffer2, n2 * sizeof(int64_t));
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
                        (partitions1[j] - prevPartitionEnd1) * sizeof(int64_t));
            std::memcpy(keys2 + prevPartitionEnd2, buffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(T2));
            std::memcpy(indexes2 + prevPartitionEnd2, indexesBuffer2 + prevPartitionEnd2,
                        (partitions2[j] - prevPartitionEnd2) * sizeof(int64_t));
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
  inline void processMicroBatch(uint32_t microBatchStart, uint32_t microBatchSize, uint32_t& i,
                                uint32_t n, const U1* keys, U1* buffer, const int64_t* indexes,
                                int64_t* indexesBuffer, std::vector<int>& buckets,
                                std::vector<int>& partitions, uint32_t& shifts, uint32_t& mask,
                                uint32_t& radixBits, uint32_t& numBuckets, uint32_t& i_2,
                                uint32_t n_2, const U2* keys_2, U2* buffer_2,
                                const int64_t* indexes_2, int64_t* indexesBuffer_2,
                                std::vector<int>& buckets_2, std::vector<int>& partitions_2) {
    eventSet.readCounters();

    for(; i < microBatchStart + microBatchSize; i++) { // Run chunk
      auto index = buckets[(keys[i] >> shifts) & mask]++;
      buffer[index] = keys[i];
      indexesBuffer[index] = indexes[i];
    }

    eventSet.readCountersAndUpdateDiff();

    // This calls the Output Transformer
    if(microBatchSize == tuplesPerHazardCheck &&
       monitor.robustnessIncreaseRequired(microBatchSize)) {
      --radixBits;
      ++shifts;
      numBuckets >>= 1U;
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

  // This function represents the Output Transformer
  template <typename U>
  inline void mergePartitions(U* buffer, int64_t* indexesBuffer, std::vector<int>& partitions,
                              std::vector<int>& buckets, uint32_t numBuckets, bool valuesInBuffer) {
    if(valuesInBuffer) {                         // Skip if no elements have been scattered yet
      for(uint32_t j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1U];
        auto srcIndex = partitions[j << 1U];
        auto numElements = buckets[(j << 1U) + 1] - srcIndex;
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int64_t));
      }
    }

    for(uint32_t j = 0; j < numBuckets; ++j) { // Merge histogram values
      buckets[j] = buckets[j << 1U] + (buckets[(j << 1U) + 1] - partitions[j << 1U]);
    }

    for(uint32_t j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j - 1] = partitions[(j << 1U) - 1];
    }
    partitions.resize(numBuckets);
  }

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  uint32_t tuplesPerHazardCheck;
  std::vector<T1> keysTmpBuffer1;
  std::vector<int64_t> indexesTmpBuffer1;
  std::vector<T2> keysTmpBuffer2;
  std::vector<int64_t> indexesTmpBuffer2;

  uint32_t nInput1{};
  T1* keysInput1{};
  int64_t* indexesInput1{};
  std::vector<int> buckets1;
  vectorOfPairs<int, int> outputPartitions1;
  int overallOffset1{};

  uint32_t nInput2{};
  T2* keysInput2{};
  int64_t* indexesInput2{};
  std::vector<int> buckets2;
  vectorOfPairs<int, int> outputPartitions2;
  int overallOffset2{};

  uint32_t minimumRadixBits{};
  uint32_t radixBitsOperator{};
  uint32_t msbToPartitionInput{};
  int maxElementsPerPartition{};
  std::atomic<int>* totalOutputPartitions{};
};

template <typename T1, typename T2>
static PartitionAdaptiveRawArrays<T1, T2>& getPartitionAdaptiveRawArrays() {
  thread_local static PartitionAdaptiveRawArrays<T1, T2> partitionAdaptiveRawArrays;
  return partitionAdaptiveRawArrays;
}

/************************************ MULTI-THREADED ***********************************/

// This class acts as the Dispatcher
template <typename T1, typename T2> class PartitionAdaptiveParallelFirstPass {
public:
  PartitionAdaptiveParallelFirstPass(const ExpressionSpanArguments& keySpans1_,
                                     const ExpressionSpanArguments& keySpans2_, int batchNumStart1_,
                                     int batchNumEnd1_, uint32_t n1, int batchNumStart2_,
                                     int batchNumEnd2_, uint32_t n2, uint32_t minimumRadixBits_,
                                     uint32_t startingRadixBits,
                                     std::atomic<uint32_t>& globalRadixBits_,
                                     uint32_t msbToPartitionInput_, int maxElementsPerPartition_,
                                     std::atomic<int>& threadsStillRunning_)
      : nInput1(n1), keySpans1(keySpans1_), batchNumStart1(batchNumStart1_),
        batchNumEnd1(batchNumEnd1_), nInput2(n2), keySpans2(keySpans2_),
        batchNumStart2(batchNumStart2_), batchNumEnd2(batchNumEnd2_),
        minimumRadixBits(minimumRadixBits_), radixBits(startingRadixBits),
        globalRadixBits(globalRadixBits_), msbToPartitionInput(msbToPartitionInput_),
        maxElementsPerPartition(maxElementsPerPartition_),
        threadsStillRunning(threadsStillRunning_), eventSet(getThreadEventSet()),
        monitor(MonitorPartition(eventSet.getCounterDiffsPtr() + EVENT::DTLB_STORE_MISSES)),
        tuplesPerHazardCheck(partitionConfig::TUPLES_PER_HAZARD_CHECK) {

#ifdef DEBUG
    std::cout << "For this thread: n1: " << nInput1U << ", n2: " << nInput2 << std::endl;
#endif

    buckets1 = std::vector<int>(1 + (1U << radixBits), 0);
    returnBuffer1 = std::make_shared_for_overwrite<std::remove_cv_t<T1>[]>(nInput1); // NOLINT
    returnIndexes1 = std::make_shared_for_overwrite<int64_t[]>(nInput1);
    inputIndexes1 = std::make_unique_for_overwrite<int64_t[]>(nInput1);

    auto keySpansStart1 = keySpans1.begin();
    std::advance(keySpansStart1, batchNumStart1);
    auto keySpansEnd1 = keySpans1.begin();
    std::advance(keySpansEnd1, batchNumEnd1);

    auto* indexesPtr1 = inputIndexes1.get();
    uint64_t spanNumber = batchNumStart1;
    uint32_t indexNumber = 0;
    for(auto spansIt = keySpansStart1; spansIt != keySpansEnd1; ++spansIt) {
      uint32_t spanOffset = 0;
      for(size_t i = 0; i < std::get<Span<T1>>(*spansIt).size(); ++i) {
        indexesPtr1[indexNumber++] =
            static_cast<int64_t>((spanNumber << partitionConfig::SPAN_OFFSET_BITS) | spanOffset++);
      }
      spanNumber++;
    }

    buckets2 = std::vector<int>(1 + (1U << radixBits), 0);
    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    returnIndexes2 = std::make_shared_for_overwrite<int64_t[]>(nInput2);
    inputIndexes2 = std::make_unique_for_overwrite<int64_t[]>(nInput2);

    auto keySpansStart2 = keySpans2.begin();
    std::advance(keySpansStart2, batchNumStart2);
    auto keySpansEnd2 = keySpans2.begin();
    std::advance(keySpansEnd2, batchNumEnd2);

    auto* indexesPtr2 = inputIndexes2.get();
    spanNumber = batchNumStart2;
    indexNumber = 0;
    for(auto spansIt = keySpansStart2; spansIt != keySpansEnd2; ++spansIt) {
      uint32_t spanOffset = 0;
      for(size_t i = 0; i < std::get<Span<T2>>(*spansIt).size(); ++i) {
        indexesPtr2[indexNumber++] =
            static_cast<int64_t>((spanNumber << partitionConfig::SPAN_OFFSET_BITS) | spanOffset++);
      }
      spanNumber++;
    }

#ifdef DEBUG
    std::cout << "Initial indexes1: ";
    printArray<int64_t>(indexesPtr1, nInput1);
    std::cout << "Initial indexes2: ";
    printArray<int64_t>(indexesPtr2, nInput2);
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
    printArray<int>(partitions1.data(), partitions1.size());
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T1>>(returnBuffer1.get(), nInput1);
    std::cout << "Indexes: " << std::endl;
    printArray<int64_t>(returnIndexes1.get(), nInput1);

    std::cout << "Table 2 results: " << std::endl;
    std::cout << "partitions2.size(): " << partitions2.size() << std::endl;
    std::cout << "partition2Positions: " << std::endl;
    printArray<int>(partitions2.data(), partitions2.size());
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T2>>(returnBuffer2.get(), nInput2);
    std::cout << "Indexes: " << std::endl;
    printArray<int64_t>(returnIndexes2.get(), nInput2);
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
    auto buffer1 = returnBuffer1.get(); // Output Collector
    auto* indexes1 = inputIndexes1.get();
    auto* indexesBuffer1 = returnIndexes1.get();
    auto* buffer2 = returnBuffer2.get(); // Output Collector
    auto* indexes2 = inputIndexes2.get();
    auto* indexesBuffer2 = returnIndexes2.get();

    uint32_t shifts = msbToPartitionInput - radixBits;
    uint32_t numBuckets = 1U << radixBits;
    uint32_t mask = numBuckets - 1;

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
    std::cout << "batchNumStart1: " << batchNumStart1U << ", batchNumEnd1: " << batchNumEnd1
              << ", batchNumStart2: " << batchNumStart2 << ", batchNumEnd2: " << batchNumEnd2
              << std::endl;
#endif

    // Complete histogram for array 1
    uint32_t i = 0;
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

    bool keysComplete1 = false;
    bool keysComplete2 = false;
    auto outerIterator1 = keySpansStart1;
    auto innerIterator1 = (std::get<Span<T1>>(*outerIterator1)).begin();
    auto outerIterator2 = keySpansStart2;
    auto innerIterator2 = (std::get<Span<T2>>(*outerIterator2)).begin();
    int offset1 = 0;
    int offset2 = 0;

    // This loop combined with the outer and inner iterators acts as the Input Interleaver
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

    // This is the Synchroniser
    threadsStillRunning.fetch_sub(1);
    if(radixBits > minimumRadixBits) { // Final synchronisation required with all complete threads
      while(threadsStillRunning > 0) {
        /* busy wait */
      }
      uint32_t finalGlobalRadixBits = globalRadixBits.load();
      while(radixBits != finalGlobalRadixBits) {
        --radixBits;
        numBuckets >>= 1U;
        mergePartitions(buffer1, indexesBuffer1, partitions1, buckets1, numBuckets, true);
        mergePartitions(buffer2, indexesBuffer2, partitions2, buckets2, numBuckets, true);
      }
    }
  }

  template <typename U1, typename U2, typename OuterIt, typename InnerIt, typename OuterIt_2,
            typename InnerIt_2>
  inline void processSpansMicroBatch(
      OuterIt& outerIterator, OuterIt& outerIteratorEnd, InnerIt& innerIterator,
      std::remove_cv_t<U1>* buffer, const int64_t* indexes, int64_t* indexesBuffer,
      std::vector<int>& buckets, std::vector<int>& partitions, int& offset, bool& keysComplete,
      uint32_t& shifts, uint32_t& mask, uint32_t& numBuckets, OuterIt_2& outerIterator_2,
      OuterIt_2& outerIteratorEnd_2, InnerIt_2& innerIterator_2, std::remove_cv_t<U2>* buffer_2,
      const int64_t* indexes_2, int64_t* indexesBuffer_2, std::vector<int>& buckets_2,
      std::vector<int>& partitions_2, int& offset_2, bool& keysComplete_2) {
    uint32_t processed = 0;
    uint32_t microBatchChunkSize = 0;
    eventSet.readCounters();
    while(processed < tuplesPerHazardCheck) {
      microBatchChunkSize = std::min(
          tuplesPerHazardCheck - processed,
          static_cast<uint32_t>((std::get<Span<U1>>(*outerIterator)).end() - innerIterator));
      for(auto i = 0U; i < microBatchChunkSize; ++i) { // Run chunk
        auto index = buckets[((*innerIterator) >> shifts) & mask]++;
        buffer[index] = *(innerIterator++);
        indexesBuffer[index] = indexes[offset + processed++];
      }
      if(innerIterator == (std::get<Span<U1>>(*outerIterator)).end()) {
        if(++outerIterator == outerIteratorEnd) {
          keysComplete = true;
          break;
        } else { // NOLINT
          innerIterator = (std::get<Span<U1>>(*outerIterator)).begin();
        }
      }
    }
    eventSet.readCountersAndUpdateDiff();
    offset += processed;

    // This is the Synchroniser
    uint32_t tmpGlobalRadixBits; // NOLINT
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
      numBuckets >>= 1U;
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
            } else { // NOLINT
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
            } else { // NOLINT
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

  // This function represents the Output Transformer
  template <typename U>
  inline void mergePartitions(U* buffer, int64_t* indexesBuffer, std::vector<int>& partitions,
                              std::vector<int>& buckets, uint32_t numBuckets, bool valuesInBuffer) {
    if(valuesInBuffer) {                         // Skip if no elements have been scattered yet
      for(uint32_t j = 0; j < numBuckets; ++j) { // Move values in buffer
        auto destIndex = buckets[j << 1U];
        auto srcIndex = partitions[(j << 1U) + 1];
        auto numElements = buckets[(j << 1U) + 1] - srcIndex;
        std::memmove(&buffer[destIndex], &buffer[srcIndex], numElements * sizeof(U)); // May overlap
        std::memmove(&indexesBuffer[destIndex], &indexesBuffer[srcIndex],
                     numElements * sizeof(int64_t));
      }
    }

    for(uint32_t j = 0; j < numBuckets; ++j) { // Merge histogram values
      buckets[j] = buckets[j << 1U] + (buckets[(j << 1U) + 1] - partitions[(j << 1U) + 1]);
    }

    for(uint32_t j = 1; j <= numBuckets; ++j) { // Merge partitions and reduce size
      partitions[j] = partitions[j << 1U];
    }
    partitions.resize(numBuckets + 1);
  }

  uint32_t nInput1;
  const ExpressionSpanArguments& keySpans1;
  int batchNumStart1;
  int batchNumEnd1;
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::unique_ptr<std::remove_cv_t<T1>[]> tmpBuffer1;
  std::shared_ptr<int64_t[]> returnIndexes1;
  std::unique_ptr<int64_t[]> inputIndexes1;
  std::vector<int> buckets1;
  std::vector<int> partitions1;

  uint32_t nInput2;
  const ExpressionSpanArguments& keySpans2;
  int batchNumStart2;
  int batchNumEnd2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::unique_ptr<std::remove_cv_t<T2>[]> tmpBuffer2;
  std::shared_ptr<int64_t[]> returnIndexes2;
  std::unique_ptr<int64_t[]> inputIndexes2;
  std::vector<int> buckets2;
  std::vector<int> partitions2;

  uint32_t minimumRadixBits;
  uint32_t radixBits;
  std::atomic<uint32_t>& globalRadixBits;
  uint32_t msbToPartitionInput;
  int maxElementsPerPartition;
  std::atomic<int>& threadsStillRunning;

  PAPI_eventSet& eventSet;
  MonitorPartition monitor;
  uint32_t tuplesPerHazardCheck;
};

template <typename T1, typename T2> class PartitionAdaptiveParallel {
public:
  PartitionAdaptiveParallel(const ExpressionSpanArguments& keySpans1_,
                            const ExpressionSpanArguments& keySpans2_, int dop_)
      : nInput1(0), keySpans1(keySpans1_), nInput2(0), keySpans2(keySpans2_), dop(dop_),
        threadPool(ThreadPool::getInstance(dop_)), synchroniser(Synchroniser::getInstance()),
        totalOutputPartitions(0) {
    assert(threadPool.getNumThreads() >= dop); // Will have a deadlock otherwise

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
      int msb1, msb2; // NOLINT
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
    returnIndexes1 = std::make_shared_for_overwrite<int64_t[]>(nInput1);

    returnBuffer2 = std::make_shared_for_overwrite<std::remove_cv_t<T2>[]>(nInput2);
    returnIndexes2 = std::make_shared_for_overwrite<int64_t[]>(nInput2);
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

    uint32_t numBatches1 = keySpans1.size();
    uint32_t numBatches2 = keySpans2.size();
    int firstPassDop = convertToValidDopValue(
        std::min(std::min(numBatches1, numBatches2), static_cast<uint32_t>(dop)));
    threadsStillRunning = firstPassDop;

#ifdef DEBUG
    std::cout << "firstPassDop: " << firstPassDop << std::endl;
    std::cout << "numBatches1: " << numBatches1U << std::endl;
    std::cout << "numBatches2: " << numBatches2 << std::endl;
#endif

    int baselineBatchesPerThread1 = numBatches1 / firstPassDop;
    int baselineBatchesPerThread2 = numBatches2 / firstPassDop;
    int remainingBatches1 = numBatches1 % firstPassDop;
    int remainingBatches2 = numBatches2 % firstPassDop;
    int startBatchNum1 = 0;
    int startBatchNum2 = 0;
    int endBatchNum1, endBatchNum2;           // NOLINT
    int batchesPerThread1, batchesPerThread2; // NOLINT
    int overallOffset1, overallOffset2;       // NOLINT
    int elementsInThread1, elementsInThread2; // NOLINT

    std::vector<PartitionedArrayAlt<std::remove_cv_t<T1>>> firstPassPartitions1(firstPassDop);
    std::vector<PartitionedArrayAlt<std::remove_cv_t<T2>>> firstPassPartitions2(firstPassDop);

    for(auto taskNum = 0; taskNum < firstPassDop; ++taskNum) {
      if(taskNum < firstPassDop - 1) {
        batchesPerThread1 =
            baselineBatchesPerThread1 + static_cast<int>(taskNum < remainingBatches1);
        batchesPerThread2 =
            baselineBatchesPerThread2 + static_cast<int>(taskNum < remainingBatches2);
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

      threadPool.enqueue([this, startBatchNum1, endBatchNum1, elementsInThread1, startBatchNum2,
                          endBatchNum2, elementsInThread2, taskNum, &firstPassPartitions1,
                          &firstPassPartitions2] {
        auto op = PartitionAdaptiveParallelFirstPass<std::remove_cv_t<T1>, std::remove_cv_t<T2>>(
            keySpans1, keySpans2, startBatchNum1, endBatchNum1, elementsInThread1, startBatchNum2,
            endBatchNum2, elementsInThread2, minimumRadixBits, startingRadixBits, globalRadixBits,
            msbToPartitionInput, maxElementsPerPartition, threadsStillRunning);
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

    std::vector<int> partitions1;
    std::vector<int> partitions2;
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
    printArray<int64_t>(returnIndexes1.get(), nInput1);

    std::cout << "Merged results from table2 first pass: " << std::endl;
    std::cout << "Partitions: " << std::endl;
    printArray<int>(partitions2.data(), static_cast<int>(partitions2.size()));
    std::cout << "Keys: " << std::endl;
    printArray<std::remove_cv_t<T2>>(returnBuffer2.get(), nInput2);
    std::cout << "Indexes: " << std::endl;
    printArray<int64_t>(returnIndexes2.get(), nInput2);
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
          auto* tableOnePartitionPositions =
              furtherPartitioningResults[j]->tableOnePartitionPositions.get();
          auto* tableTwoPartitionPositions =
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
                              T* keyBuffer, int64_t* indexesBuffer) {
    std::vector<int> mergedPartitions;
    uint32_t numPartitions = partitionedArrays[0].partitionPositions->size() - 1;
    mergedPartitions.reserve(numPartitions);

    int totalElements = 0;
    for(uint32_t partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
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
        printArray<int64_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
                            threadPartitionElements);
#endif
        memcpy(keyBuffer + totalElements,
               &(partitionedArrays[threadNum].partitionedKeys[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(T));
        memcpy(indexesBuffer + totalElements,
               &(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(int64_t));

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
    uint32_t numPartitions = partitionedArrays[0].partitionPositions->size() - 1;
    mergedPartitions.reserve(numPartitions);

    int totalElements = 0;
    for(uint32_t partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
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
        printArray<int64_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
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
                    int64_t* indexesBuffer) {
    uint32_t numPartitions = partitionedArrays[0].partitionPositions->size() - 1;

    int totalElements = 0;
    for(uint32_t partitionNum = 0; partitionNum < numPartitions; ++partitionNum) {
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
        printArray<int64_t>(&(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
                            threadPartitionElements);
#endif
        memcpy(indexesBuffer + totalElements,
               &(partitionedArrays[threadNum].indexes[threadPartitionElementsStart]),
               threadPartitionElements * sizeof(int64_t));

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

  template <typename U> inline uint32_t getMsb(const ExpressionSpanArguments& keySpans, uint32_t& n) {
    auto largest = std::numeric_limits<U>::min();
    for(const auto& untypedSpan : keySpans) {
      auto& span = std::get<Span<U>>(untypedSpan);
      n += span.size();
      for(auto& key : span) {
        largest = std::max(largest, key);
      }
    }

    uint32_t msbToPartition = 0;
    while(largest != 0) {
      largest >>= 1;
      msbToPartition++;
    }
    return msbToPartition;
  }

  uint32_t nInput1;
  const ExpressionSpanArguments& keySpans1;
  std::shared_ptr<std::remove_cv_t<T1>[]> returnBuffer1;
  std::shared_ptr<int64_t[]> returnIndexes1;
  vectorOfPairs<int, int> outputPartitions1;

  uint32_t nInput2;
  const ExpressionSpanArguments& keySpans2;
  std::shared_ptr<std::remove_cv_t<T2>[]> returnBuffer2;
  std::shared_ptr<int64_t[]> returnIndexes2;
  vectorOfPairs<int, int> outputPartitions2;

  int dop;
  uint32_t minimumRadixBits;
  uint32_t startingRadixBits;
  std::atomic<uint32_t> globalRadixBits;
  uint32_t msbToPartitionInput{};
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

  std::vector<ExpressionSpanArguments> partitionsOfKeySpans1, partitionsOfIndexSpans1; // NOLINT
  std::vector<ExpressionSpanArguments> partitionsOfKeySpans2, partitionsOfIndexSpans2; // NOLINT
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

    while(partitionSize1 < config::minPartitionSize && partitionNum < partitions1.size()) {
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
        Span<int64_t>(indexes1 + partitionStart1, partitionSize1,
                      [ptr = partitionedTables.partitionedArrayOne.indexes]() {}));

    partitionsOfKeySpans2.emplace_back(
        Span<T2>(keys2 + partitionStart2, partitionSize2,
                 [ptr = partitionedTables.partitionedArrayTwo.partitionedKeys]() {}));
    partitionsOfIndexSpans2.emplace_back(
        Span<int64_t>(indexes2 + partitionStart2, partitionSize2,
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
    printArray<int64_t>(indexes, size);
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
        outputIndexSpans.emplace_back(Span<int64_t>());
      } else {
        outputKeySpans.emplace_back(
            Span<std::remove_cv_t<T>>(keys + prevSpanEndIndex, index - prevSpanEndIndex,
                                      [ptr = partitionedArray.partitionedKeys]() {}));
        outputIndexSpans.emplace_back(Span<int64_t>(indexes + prevSpanEndIndex,
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
    outputIndexSpans.emplace_back(Span<int64_t>(indexes + partition.first, partition.second,
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
        }
        if(partitionImplementation == PartitionOperators::RadixBitsFixedMax) {
          std::string name = "Partition_startRadixBits";
          auto radixBitsMax =
              static_cast<int>(MachineConstants::getInstance().getMachineConstant(name));
          auto partitionOperator = Partition<T1, T2>(tableOneKeys, tableTwoKeys, radixBitsMax);
          return partitionOperator.processInput();
        }
        if(partitionImplementation == PartitionOperators::RadixBitsAdaptive) {
          auto partitionOperator = PartitionAdaptive<T1, T2>(tableOneKeys, tableTwoKeys);
          return partitionOperator.processInput();
        }
        throw std::runtime_error("Invalid selection of 'Partition' implementation!");
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
    auto& synchroniser = Synchroniser::getInstance();
    ThreadPool::getInstance(2).enqueue([&tableOneKeys, &partitionedTables,
                                        &tableOnePartitionsOfKeySpans,
                                        &tableOnePartitionsOfIndexSpans, &synchroniser] {
      auto [tableOnePartitionsOfKeySpansTmp, tableOnePartitionsOfIndexSpansTmp] =
          createPartitionsOfSpansAlignedToTableBatches<T1>(partitionedTables.partitionedArrayOne,
                                                           tableOneKeys);
      tableOnePartitionsOfKeySpans = std::move(tableOnePartitionsOfKeySpansTmp);
      tableOnePartitionsOfIndexSpans = std::move(tableOnePartitionsOfIndexSpansTmp);
      synchroniser.taskComplete();
    });
    ThreadPool::getInstance(2).enqueue([&tableTwoKeys, &partitionedTables,
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

// NOLINTEND(hicpp-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays,
// readability-function-cognitive-complexity)