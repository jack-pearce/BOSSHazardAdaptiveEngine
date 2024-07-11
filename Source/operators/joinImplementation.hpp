#ifndef BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP

#include <memory>
#include <tsl/robin_map.h>
#include <type_traits>
#include <vector>
#include <algorithm>

#include "HazardAdaptiveEngine.hpp"
#include "config.hpp"

namespace adaptive {

using config::minPartitionSize;

// This Join has already been partitioned and so can be part of the vectorized pipeline.
// Therefore it only creates single spans.
template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2, const Span<int64_t>& indexes1,
                       const Span<int64_t>& indexes2) {
  static_assert(std::is_integral<T1>::value, "Join key column must be an integer type");
  static_assert(std::is_integral<T2>::value, "Join key column must be an integer type");

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
template <typename T1, typename T2>
JoinResultIndexes join(const ExpressionSpanArguments& keySpans1,
                       const ExpressionSpanArguments& keySpans2) {
  static_assert(std::is_integral<T1>::value, "Join key column must be an integer type");
  static_assert(std::is_integral<T2>::value, "Join key column must be an integer type");

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

  int64_t spanNumber = 0;
  for(const auto& untypedSpan : keySpans1) {
    const auto& span = std::get<Span<T1>>(untypedSpan);
    uint32_t spanOffset = 0;
    for(const auto& key : span) {
      it = map.find(std::remove_cv_t<T1>(key));
      if(it != map.end()) {
        it.value().push_back((spanNumber << 32) | spanOffset++);
      } else {
        map.insert({key, {(spanNumber << 32) | spanOffset++}});
      }
    }
    spanNumber++;
  }

  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr1 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr1->reserve(std::max(n1, n2));
  std::shared_ptr<std::vector<int64_t>> resultIndexesPtr2 =
      std::make_shared<std::vector<int64_t>>();
  resultIndexesPtr2->reserve(std::max(n1, n2));

  spanNumber = 0;
  for(const auto& untypedSpan : keySpans2) {
    const auto& span = std::get<Span<T2>>(untypedSpan);
    uint32_t spanOffset = 0;
    for(const auto& key : span) {
      it = map.find(static_cast<std::remove_cv_t<T1>>(key));
      if(it != map.end()) {
        for(const auto& matchedIndex : it.value()) {
          resultIndexesPtr1->push_back(matchedIndex);
          resultIndexesPtr2->push_back((spanNumber << 32) | spanOffset);
        }
      }
      spanOffset++;
    }
    spanNumber++;
  }

  size_t numResultSpans = (resultIndexesPtr1->size() + minPartitionSize - 1) / minPartitionSize;
  ExpressionSpanArguments resultSpans1;
  ExpressionSpanArguments resultSpans2;
  resultSpans1.reserve(numResultSpans);
  resultSpans2.reserve(numResultSpans);

  size_t spanStart = 0;
  size_t spanSize = minPartitionSize;
  for(int i = 0; i < std::max(static_cast<int>(numResultSpans) - 1,0); ++i) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
    spanStart += spanSize;
  }
  spanSize = resultIndexesPtr1->size() - spanStart;
  if (spanSize > 0) {
    resultSpans1.emplace_back(Span<int64_t>(resultIndexesPtr1->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr1]() {}));
    resultSpans2.emplace_back(Span<int64_t>(resultIndexesPtr2->data() + spanStart, spanSize,
                                            [ptr = resultIndexesPtr2]() {}));
  }

  return {std::move(resultSpans1), std::move(resultSpans2)};
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_JOINIMPLEMENTATION_HPP
