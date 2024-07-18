#ifndef BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <tsl/robin_map.h>
#include <type_traits>
#include <vector>

#include "HazardAdaptiveEngine.hpp"
#include "config.hpp"
#include "utilities/memory.hpp"

namespace adaptive {

using config::minPartitionSize;

/*********************************** SINGLE KEY JOINS *************************************/

// This Join has already been partitioned and so can be part of the vectorized pipeline.
// Therefore it only creates single spans. This Join is parallelised by the vectorization
// coordinator engine so a single-threaded implementation is only required here.
template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2, const Span<int64_t>& indexes1,
                       const Span<int64_t>& indexes2) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value,
                "Join key column must be an integer type");

  size_t n1 = 0;
  for(const auto& untypedSpan : keySpans1) {
    const auto& span = std::get<Span<T1>>(untypedSpan);
    n1 += span.size();
  }
  assert(n1 == indexes1.size());

  size_t n2 = 0;
  for(const auto& untypedSpan : keySpans2) {
    const auto& span = std::get<Span<T2>>(untypedSpan);
    n2 += span.size();
  }
  assert(n2 == indexes2.size());

  tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>> map(static_cast<int>(n1));
  typename tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>>::iterator it;

  size_t indexNum = 0;
  for(const auto& untypedSpan : keySpans1) {
    const auto& span = std::get<Span<T1>>(untypedSpan);
    for(const auto& key : span) {
      it = map.find(std::remove_cv_t<T1>(key));
      if(it != map.end()) {
        it.value().push_back(indexes1[indexNum++]);
      } else {
        map.insert({key, {indexes1[indexNum++]}});
      }
    }
  }

  std::vector<int64_t> resultIndexes1;
  resultIndexes1.reserve(std::max(n1, n2));
  std::vector<int64_t> resultIndexes2;
  resultIndexes2.reserve(std::max(n1, n2));

  indexNum = 0;
  for(const auto& untypedSpan : keySpans2) {
    const auto& span = std::get<Span<T2>>(untypedSpan);
    for(const auto& key : span) {
      it = map.find(static_cast<T1>(key));
      if(it != map.end()) {
        for(const auto& matchedIndex : it.value()) {
          resultIndexes1.push_back(matchedIndex);
          resultIndexes2.push_back(indexes2[indexNum]);
        }
      }
      indexNum++;
    }
  }

  ExpressionSpanArguments resultSpans1;
  ExpressionSpanArguments resultSpans2;
  resultSpans1.reserve(1);
  resultSpans2.reserve(1);

  resultSpans1.emplace_back(Span<int64_t>(std::move(resultIndexes1)));
  resultSpans2.emplace_back(Span<int64_t>(std::move(resultIndexes2)));

  return {std::move(resultSpans1), std::move(resultSpans2)};
}

// This Join has not been partitioned and so will be batch evaluated.
// Therefore it creates multiple spans so that the following operators can be vectorized.
// Since it is batch evaluated it is parallelised here.
template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2, int dop) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value,
                "Join key column must be an integer type");
  if(dop == 1) { // Single-threaded implementation

    size_t n1 = 0;
    for(const auto& untypedSpan : keySpans1) {
      const auto& span = std::get<Span<T1>>(untypedSpan);
      n1 += span.size();
    }
    size_t n2 = 0;
    for(const auto& untypedSpan : keySpans2) {
      const auto& span = std::get<Span<T2>>(untypedSpan);
      n2 += span.size();
    }

    tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>> map(static_cast<int>(n1));
    typename tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>>::iterator it;

    for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpans1.size()); spanNumber++) {
      const auto& span = std::get<Span<T1>>(keySpans1.at(spanNumber));
      for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(span.size()); spanOffset++) {
        auto key = static_cast<std::remove_cv_t<T1>>(span[spanOffset]);
        it = map.find(key);
        if(it != map.end()) {
          it.value().push_back((spanNumber << 32) | spanOffset);
        } else {
          map.insert({key, {(spanNumber << 32) | spanOffset}});
        }
      }
    }

    std::shared_ptr<std::vector<int64_t>> resultIndexesPtr1 =
        std::make_shared<std::vector<int64_t>>();
    resultIndexesPtr1->reserve(std::max(n1, n2));
    std::shared_ptr<std::vector<int64_t>> resultIndexesPtr2 =
        std::make_shared<std::vector<int64_t>>();
    resultIndexesPtr2->reserve(std::max(n1, n2));

    for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpans2.size()); spanNumber++) {
      const auto& span = std::get<Span<T2>>(keySpans2.at(spanNumber));
      for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(span.size()); spanOffset++) {
        auto key = static_cast<std::remove_cv_t<T1>>(span[spanOffset]);
        it = map.find(key);
        if(it != map.end()) {
          for(const auto& matchedIndex : it.value()) {
            resultIndexesPtr1->push_back(matchedIndex);
            resultIndexesPtr2->push_back((spanNumber << 32) | spanOffset);
          }
        }
      }
    }

    size_t numResultSpans = (resultIndexesPtr1->size() + minPartitionSize - 1) / minPartitionSize;
    ExpressionSpanArguments resultSpans1;
    ExpressionSpanArguments resultSpans2;
    resultSpans1.reserve(numResultSpans);
    resultSpans2.reserve(numResultSpans);

    size_t spanStart = 0;
    size_t spanSize = minPartitionSize;
    for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
      resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr1]() {}));
      resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr2]() {}));
      spanStart += spanSize;
    }
    spanSize = resultIndexesPtr1->size() - spanStart;
    if(spanSize > 0) {
      resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr1]() {}));
      resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr2]() {}));
    }

    return {std::move(resultSpans1), std::move(resultSpans2)};
  }

  // Multi-threaded implementation

  auto& threadPool = ThreadPool::getInstance(dop);
  assert(threadPool.getNumThreads() >= dop);
  auto& synchroniser = Synchroniser::getInstance();

  size_t n1 = 0;
  size_t n2 = 0;

  threadPool.enqueue([&synchroniser, &keySpans1, &n1] {
    for(const auto& untypedSpan : keySpans1) {
      const auto& span = std::get<Span<T1>>(untypedSpan);
      n1 += span.size();
    }
    synchroniser.taskComplete();
  });
  threadPool.enqueue([&synchroniser, &keySpans2, &n2] {
    for(const auto& untypedSpan : keySpans2) {
      const auto& span = std::get<Span<T2>>(untypedSpan);
      n2 += span.size();
    }
    synchroniser.taskComplete();
  });
  synchroniser.waitUntilComplete(2);

  tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>> map(static_cast<int>(n1));
  typename tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>>::iterator it;

  for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpans1.size()); spanNumber++) {
    const auto& span = std::get<Span<T1>>(keySpans1.at(spanNumber));
    for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(span.size()); spanOffset++) {
      auto key = static_cast<std::remove_cv_t<T1>>(span[spanOffset]);
      it = map.find(key);
      if(it != map.end()) {
        it.value().push_back((spanNumber << 32) | spanOffset);
      } else {
        map.insert({key, {(spanNumber << 32) | spanOffset}});
      }
    }
  }

  int numSpans = static_cast<int>(keySpans2.size());
  dop = std::min(dop, numSpans);
  int avgExpectedResultSizePerThread = static_cast<int>(std::max(n1, n2)) / dop;
  int spansPerThreadBaseline = numSpans / dop;
  int remainingSpans = numSpans % dop;
  int64_t startSpan = 0;
  int spansPerThread; // NOLINT

  std::vector<std::vector<int64_t>> results1(dop);
  std::vector<std::vector<int64_t>> results2(dop);
  std::mutex results1Mutex;
  std::mutex results2Mutex;

  for(int taskNum = 0; taskNum < dop; ++taskNum) {
    spansPerThread = spansPerThreadBaseline + (taskNum < remainingSpans);

    threadPool.enqueue([startSpan, spansPerThread, avgExpectedResultSizePerThread, taskNum,
                        &keySpans2, &map, &results1, &results2, &results1Mutex, &results2Mutex,
                        &synchroniser] {
      std::vector<int64_t> resultsIndexes1;
      resultsIndexes1.reserve(avgExpectedResultSizePerThread);
      std::vector<int64_t> resultsIndexes2;
      resultsIndexes2.reserve(avgExpectedResultSizePerThread);

      typename tsl::robin_map<std::remove_cv_t<T1>, std::vector<int64_t>>::iterator it;

      for(int64_t spanNumber = startSpan; spanNumber < startSpan + spansPerThread; spanNumber++) {
        const auto& span = std::get<Span<T2>>(keySpans2.at(spanNumber));
        for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(span.size());
            spanOffset++) {
          auto key = static_cast<std::remove_cv_t<T1>>(span[spanOffset]);
          it = map.find(key);
          if(it != map.end()) {
            for(const auto& matchedIndex : it.value()) {
              resultsIndexes1.push_back(matchedIndex);
              resultsIndexes2.push_back((spanNumber << 32) | spanOffset);
            }
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(results1Mutex);
        results1[taskNum] = std::move(resultsIndexes1);
      }
      {
        std::lock_guard<std::mutex> lock(results2Mutex);
        results2[taskNum] = std::move(resultsIndexes2);
      }
      synchroniser.taskComplete();
    });

    startSpan += static_cast<int64_t>(spansPerThread);
  }
  synchroniser.waitUntilComplete(dop);

  size_t totalResultSize = 0;
  for(const auto& resultsVec : results1) {
    totalResultSize += resultsVec.size();
  }

  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr1 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr1->reserve(totalResultSize);
  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr2 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr2->reserve(totalResultSize);

  threadPool.enqueue([&synchroniser, &results1, &resultIndexesPtr1] {
    for(const auto& vec : results1) {
      resultIndexesPtr1->insert(resultIndexesPtr1->end(), vec.begin(), vec.end());
    }
    synchroniser.taskComplete();
  });
  threadPool.enqueue([&synchroniser, &results2, &resultIndexesPtr2] {
    for(const auto& vec : results2) {
      resultIndexesPtr2->insert(resultIndexesPtr2->end(), vec.begin(), vec.end());
    }
    synchroniser.taskComplete();
  });
  synchroniser.waitUntilComplete(2);

  size_t numResultSpans = (resultIndexesPtr1->size() + minPartitionSize - 1) / minPartitionSize;
  ExpressionSpanArguments resultSpans1;
  ExpressionSpanArguments resultSpans2;
  resultSpans1.reserve(numResultSpans);
  resultSpans2.reserve(numResultSpans);

  size_t spanStart = 0;
  size_t spanSize = minPartitionSize;
  for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
    spanStart += spanSize;
  }
  spanSize = resultIndexesPtr1->size() - spanStart;
  if(spanSize > 0) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
  }

  return {std::move(resultSpans1), std::move(resultSpans2)};
}

/*********************************** TWO KEY JOINS *************************************/

struct pair_hash {
  template <class T1, class T2> inline std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    // https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
    hash1 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2); // NOLINT
    return hash1;
  }
};

struct pair_equal {
  template <class T1, class T2>
  inline bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) const {
    return p1.first == p2.first && p1.second == p2.second;
  }
};

// The Partition operator only works for single key Joins, therefore a two key Join is only
// required for the batch evaluated Join since this Join has not been partitioned.
// Since this Join has not been partitioned it creates multiple spans so that the following
// operators can be vectorized. Since it is batch evaluated it is parallelised here.
template <typename T1, typename T2, typename T3, typename T4>
JoinResultIndexes join(const ExpressionSpanArguments& keySpansFirstKey1,
                       const ExpressionSpanArguments& keySpansSecondKey1,
                       const ExpressionSpanArguments& keySpansFirstKey2,
                       const ExpressionSpanArguments& keySpansSecondKey2, int dop) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value &&
                    std::is_integral<T3>::value && std::is_integral<T4>::value,
                "Join key column must be an integer type");
  if(dop == 1) { // Single-threaded implementation

    size_t n1 = 0;
    for(const auto& untypedSpan : keySpansFirstKey1) {
      const auto& span = std::get<Span<T1>>(untypedSpan);
      n1 += span.size();
    }

    size_t n2 = 0;
    for(const auto& untypedSpan : keySpansFirstKey2) {
      const auto& span = std::get<Span<T3>>(untypedSpan);
      n2 += span.size();
    }

    using Key = std::pair<int32_t, int32_t>;
    tsl::robin_map<Key, std::vector<int64_t>, pair_hash, pair_equal> map(static_cast<int>(n1));
    typename tsl::robin_map<Key, std::vector<int64_t>, pair_hash, pair_equal>::iterator it;

    for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpansFirstKey1.size());
        spanNumber++) {
      const auto& spanFirst = std::get<Span<T1>>(keySpansFirstKey1.at(spanNumber));
      const auto& spanSecond = std::get<Span<T2>>(keySpansSecondKey1.at(spanNumber));
      for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(spanFirst.size());
          spanOffset++) {
        Key key = {static_cast<int32_t>(spanFirst[spanOffset]),
                   static_cast<int32_t>(spanSecond[spanOffset])};
        it = map.find(key);
        if(it != map.end()) {
          it.value().push_back((spanNumber << 32) | spanOffset);
        } else {
          map.insert({key, {(spanNumber << 32) | spanOffset}});
        }
      }
    }

    std::shared_ptr<std::vector<int64_t>> resultIndexesPtr1 =
        std::make_shared<std::vector<int64_t>>();
    resultIndexesPtr1->reserve(std::max(n1, n2));
    std::shared_ptr<std::vector<int64_t>> resultIndexesPtr2 =
        std::make_shared<std::vector<int64_t>>();
    resultIndexesPtr2->reserve(std::max(n1, n2));

    for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpansFirstKey2.size());
        spanNumber++) {
      const auto& spanFirst = std::get<Span<T3>>(keySpansFirstKey2.at(spanNumber));
      const auto& spanSecond = std::get<Span<T4>>(keySpansSecondKey2.at(spanNumber));
      for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(spanFirst.size());
          spanOffset++) {
        Key key = {static_cast<int32_t>(spanFirst[spanOffset]),
                   static_cast<int32_t>(spanSecond[spanOffset])};
        it = map.find(key);
        if(it != map.end()) {
          for(const auto& matchedIndex : it.value()) {
            resultIndexesPtr1->push_back(matchedIndex);
            resultIndexesPtr2->push_back((spanNumber << 32) | spanOffset);
          }
        }
      }
    }

    size_t numResultSpans = (resultIndexesPtr1->size() + minPartitionSize - 1) / minPartitionSize;
    ExpressionSpanArguments resultSpans1;
    ExpressionSpanArguments resultSpans2;
    resultSpans1.reserve(numResultSpans);
    resultSpans2.reserve(numResultSpans);

    size_t spanStart = 0;
    size_t spanSize = minPartitionSize;
    for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
      resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr1]() {}));
      resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr2]() {}));
      spanStart += spanSize;
    }
    spanSize = resultIndexesPtr1->size() - spanStart;
    if(spanSize > 0) {
      resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr1]() {}));
      resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                              [ptr = resultIndexesPtr2]() {}));
    }

    return {std::move(resultSpans1), std::move(resultSpans2)};
  }

  // Multi-threaded implementation

  auto& threadPool = ThreadPool::getInstance(dop);
  assert(threadPool.getNumThreads() >= dop);
  auto& synchroniser = Synchroniser::getInstance();

  size_t n1 = 0;
  size_t n2 = 0;

  threadPool.enqueue([&synchroniser, &keySpansFirstKey1, &n1] {
    for(const auto& untypedSpan : keySpansFirstKey1) {
      const auto& span = std::get<Span<T1>>(untypedSpan);
      n1 += span.size();
    }
    synchroniser.taskComplete();
  });
  threadPool.enqueue([&synchroniser, &keySpansFirstKey2, &n2] {
    for(const auto& untypedSpan : keySpansFirstKey2) {
      const auto& span = std::get<Span<T3>>(untypedSpan);
      n2 += span.size();
    }
    synchroniser.taskComplete();
  });
  synchroniser.waitUntilComplete(2);

  using Key = std::pair<int32_t, int32_t>;
  tsl::robin_map<Key, std::vector<int64_t>, pair_hash, pair_equal> map(static_cast<int>(n1));
  typename tsl::robin_map<Key, std::vector<int64_t>, pair_hash, pair_equal>::iterator it;

  for(int64_t spanNumber = 0; spanNumber < static_cast<int64_t>(keySpansFirstKey1.size());
      spanNumber++) {
    const auto& spanFirst = std::get<Span<T1>>(keySpansFirstKey1.at(spanNumber));
    const auto& spanSecond = std::get<Span<T2>>(keySpansSecondKey1.at(spanNumber));
    for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(spanFirst.size());
        spanOffset++) {
      Key key = {static_cast<int32_t>(spanFirst[spanOffset]),
                 static_cast<int32_t>(spanSecond[spanOffset])};
      it = map.find(key);
      if(it != map.end()) {
        it.value().push_back((spanNumber << 32) | spanOffset);
      } else {
        map.insert({key, {(spanNumber << 32) | spanOffset}});
      }
    }
  }

  int numSpans = static_cast<int>(keySpansFirstKey2.size());
  dop = std::min(dop, numSpans);
  int avgExpectedResultSizePerThread = static_cast<int>(std::max(n1, n2)) / dop;
  int spansPerThreadBaseline = numSpans / dop;
  int remainingSpans = numSpans % dop;
  int64_t startSpan = 0;
  int spansPerThread; // NOLINT

  std::vector<std::vector<int64_t>> results1(dop);
  std::vector<std::vector<int64_t>> results2(dop);
  std::mutex results1Mutex;
  std::mutex results2Mutex;

  for(int taskNum = 0; taskNum < dop; ++taskNum) {
    spansPerThread = spansPerThreadBaseline + (taskNum < remainingSpans);

    threadPool.enqueue([startSpan, spansPerThread, avgExpectedResultSizePerThread, taskNum,
                        &keySpansFirstKey2, &keySpansSecondKey2, &map, &results1, &results2,
                        &results1Mutex, &results2Mutex, &synchroniser] {
      std::vector<int64_t> resultsIndexes1;
      resultsIndexes1.reserve(avgExpectedResultSizePerThread);
      std::vector<int64_t> resultsIndexes2;
      resultsIndexes2.reserve(avgExpectedResultSizePerThread);

      typename tsl::robin_map<Key, std::vector<int64_t>, pair_hash, pair_equal>::iterator it;

      for(int64_t spanNumber = startSpan; spanNumber < startSpan + spansPerThread; spanNumber++) {
        const auto& spanFirst = std::get<Span<T3>>(keySpansFirstKey2.at(spanNumber));
        const auto& spanSecond = std::get<Span<T4>>(keySpansSecondKey2.at(spanNumber));
        for(uint32_t spanOffset = 0; spanOffset < static_cast<uint32_t>(spanFirst.size());
            spanOffset++) {
          Key key = {static_cast<int32_t>(spanFirst[spanOffset]),
                     static_cast<int32_t>(spanSecond[spanOffset])};
          it = map.find(key);
          if(it != map.end()) {
            for(const auto& matchedIndex : it.value()) {
              resultsIndexes1.push_back(matchedIndex);
              resultsIndexes2.push_back((spanNumber << 32) | spanOffset);
            }
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(results1Mutex);
        results1[taskNum] = std::move(resultsIndexes1);
      }
      {
        std::lock_guard<std::mutex> lock(results2Mutex);
        results2[taskNum] = std::move(resultsIndexes2);
      }
      synchroniser.taskComplete();
    });

    startSpan += static_cast<int64_t>(spansPerThread);
  }
  synchroniser.waitUntilComplete(dop);

  size_t totalResultSize = 0;
  for(const auto& resultsVec : results1) {
    totalResultSize += resultsVec.size();
  }

  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr1 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr1->reserve(totalResultSize);
  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr2 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr2->reserve(totalResultSize);

  threadPool.enqueue([&synchroniser, &results1, &resultIndexesPtr1] {
    for(const auto& vec : results1) {
      resultIndexesPtr1->insert(resultIndexesPtr1->end(), vec.begin(), vec.end());
    }
    synchroniser.taskComplete();
  });
  threadPool.enqueue([&synchroniser, &results2, &resultIndexesPtr2] {
    for(const auto& vec : results2) {
      resultIndexesPtr2->insert(resultIndexesPtr2->end(), vec.begin(), vec.end());
    }
    synchroniser.taskComplete();
  });
  synchroniser.waitUntilComplete(2);

  size_t numResultSpans = (resultIndexesPtr1->size() + minPartitionSize - 1) / minPartitionSize;
  ExpressionSpanArguments resultSpans1;
  ExpressionSpanArguments resultSpans2;
  resultSpans1.reserve(numResultSpans);
  resultSpans2.reserve(numResultSpans);

  size_t spanStart = 0;
  size_t spanSize = minPartitionSize;
  for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1, 0); ++i) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
    spanStart += spanSize;
  }
  spanSize = resultIndexesPtr1->size() - spanStart;
  if(spanSize > 0) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
  }

  return {std::move(resultSpans1), std::move(resultSpans2)};
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP
