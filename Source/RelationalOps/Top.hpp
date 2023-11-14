#pragma once

#include "../BOSSExpressionConversions.hpp"
#include "Operator.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace boss::engines::volcano::operators {

class Top : public Operator {
public:
  Top(std::unique_ptr<Operator>&& op, int64_t n, ComplexExpression&& orderExpr)
      : input(std::move(op)), maxN(n), orderOp(toArithmeticOp(std::move(orderExpr), *input)) {
    // already build the output tuples
    auto comp = [this](auto const& lhs, auto const& rhs) { return orderOp(lhs) > orderOp(rhs); };
    while(auto candidate = input->next()) {
      output.emplace_back(std::move(*candidate));
      if(output.size() <= maxN) {
        if(output.size() == maxN) {
          // make it a heap on the inversed order, so the smallest (to pop) is always in the
          // front
          std::make_heap(output.begin(), output.end(), comp);
        }
        // if we haven't reached the max number of tuples, nothing else to do
        continue;
      }
      // push the new element, pop the smallest
      std::push_heap(output.begin(), output.end(), comp);
      std::pop_heap(output.begin(), output.end(), comp);
      output.pop_back();
    }
    if(output.size() < maxN) {
      // if we haven't reach the max number of tuples, we still need to make it a heap
      std::make_heap(output.begin(), output.end(), comp);
    }
    std::sort_heap(output.begin(), output.end(), comp);
    outputIt = output.begin();
  }

  std::optional<Tuple> next() override {
    if(outputIt == output.end()) {
      return {};
    }
    return std::move(*outputIt++);
  }

  Schema const& getSchema() const override { return input->getSchema(); }

private:
  std::unique_ptr<Operator> input;
  int64_t maxN;
  ArithmeticOp orderOp;
  std::vector<Tuple> output;
  std::vector<Tuple>::iterator outputIt;
};

} // namespace boss::engines::volcano::operators