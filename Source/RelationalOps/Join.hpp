#pragma once

#include "../BOSSExpressionConversions.hpp"
#include "Operator.hpp"
#include <memory>

namespace boss::engines::volcano::operators {

class Join : public Operator {
public:
  Join(std::unique_ptr<Operator>&& left, std::unique_ptr<Operator>&& right,
       ComplexExpression&& predExpr)
      : schema(buildSchema(left->getSchema(), right->getSchema())), leftInput(std::move(left)),
        currentLeftTuple(*leftInput->next()), predicate(toPredicate(std::move(predExpr), *this)) {
    // already build tuples from the right-side relation (cached for multiple iterations)
    while(auto rightTuple = right->next()) {
      rightTuples.emplace_back(std::move(*rightTuple));
    }
    rightTuplesIt = rightTuples.begin();
  }

  std::optional<Tuple> next() override {
    while(true) {
      if(rightTuplesIt == rightTuples.end()) {
        // rewind the right-side tuples and get the next left-side tuple
        rightTuplesIt = rightTuples.begin();
        auto nextLeftTuple = leftInput->next();
        if(!nextLeftTuple) {
          return {};
        }
        currentLeftTuple = std::move(*nextLeftTuple);
      }
      // build the next candidate with the current left-side tuple + the next right-side tuple
      auto const& currentRightTuple = *rightTuplesIt++;
      auto candidate = currentLeftTuple;
      candidate.insert(candidate.end(), currentRightTuple.begin(), currentRightTuple.end());
      // check the join condition
      if(predicate(candidate)) {
        return candidate;
      }
    }
  }

  Schema const& getSchema() const override { return schema; }

private:
  Schema schema; // new schema merging from left and right schemas
  static Schema buildSchema(Schema const& leftSchema, Schema const& rightSchema) {
    Schema schema;
    schema.insert(schema.end(), leftSchema.begin(), leftSchema.end());
    schema.insert(schema.end(), rightSchema.begin(), rightSchema.end());
    return schema;
  }
  std::unique_ptr<Operator> leftInput;
  Tuple currentLeftTuple;
  Predicate predicate;
  // for caching the right side tuples:
  std::vector<Tuple> rightTuples;
  std::vector<Tuple>::iterator rightTuplesIt;
};

} // namespace boss::engines::volcano::operators