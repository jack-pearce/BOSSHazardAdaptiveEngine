#pragma once

#include "../BOSSExpressionConversions.hpp"
#include "Operator.hpp"
#include <memory>

namespace boss::engines::volcano::operators {

class Select : public Operator {
public:
  Select(std::unique_ptr<Operator>&& op, ComplexExpression&& predExpr)
      : input(std::move(op)), predicate(toPredicate(std::move(predExpr), *input)) {}

  std::optional<Tuple> next() override {
    while(auto candidate = input->next()) {
      if(predicate(*candidate)) {
        return candidate;
      }
    }
    return {};
  }

  Schema const& getSchema() const override { return input->getSchema(); }

private:
  std::unique_ptr<Operator> input;
  Predicate predicate;
};

} // namespace boss::engines::volcano::operators