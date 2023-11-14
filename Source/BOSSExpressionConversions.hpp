#pragma once

#include "RelationalOps/Operator.hpp"
#include "Types.hpp"
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <tuple>
#include <vector>

using boss::utilities::operator""_;
using boss::expressions::generic::isComplexExpression;

namespace boss::engines::volcano {

using operators::Operator;

std::tuple<Schema, std::vector<Tuple>> toSchemaAndData(ComplexExpression&& e) {
  auto columns = std::move(e).getDynamicArguments();
  if(columns.empty()) {
    return {};
  }
  auto numTuples =
      boss::get<ComplexExpression>(boss::get<ComplexExpression>(columns[0]).getArguments()[0])
          .getArguments()
          .size();
  std::vector<Tuple> tuples(numTuples);
  Schema schema;
  for(auto&& column : columns) {
    auto [head, unused_, dynamics, unused2_] =
        boss::get<ComplexExpression>(std::move(column)).decompose();
    schema.emplace_back(std::move(head).getName());
    auto list = *std::make_move_iterator(dynamics.begin());
    auto tupleIt = tuples.begin();
    for(auto&& valExpr : boss::get<ComplexExpression>(list).getArguments()) {
      boss::expressions::generic::visit(
          [&tupleIt](auto&& val) {
            if constexpr(std::is_same_v<std::decay_t<decltype(val)>, int64_t> ||
                         std::is_same_v<std::decay_t<decltype(val)>, double_t>) {
              tupleIt->emplace_back(std::move(val));
            } else {
              throw std::runtime_error("unsupported type as a tuple value");
            }
          },
          std::move(valExpr));
      ++tupleIt;
    }
  }
  return {std::move(schema), std::move(tuples)};
}

ArithmeticOp toArithmeticOp(Expression&& e, Operator const& input);

ArithmeticOp toArithmeticOp(ComplexExpression&& e, Operator const& input) {
  if(e.getHead() == "Plus"_) {
    auto args = std::move(e).getArguments();
    return std::accumulate(
        std::make_move_iterator(args.begin() + 1), std::make_move_iterator(args.end()),
        toArithmeticOp(*args.begin(), input), [&input](auto&& acc, auto&& expr) -> ArithmeticOp {
          return [leftArg = std::move(acc), rightArg = toArithmeticOp(std::move(expr), input)](
                     Tuple const& tuple) { return leftArg(tuple) + rightArg(tuple); };
        });
  }
  if(e.getHead() == "Multiply"_) {
    auto args = std::move(e).getArguments();
    return std::accumulate(
        std::make_move_iterator(args.begin() + 1), std::make_move_iterator(args.end()),
        toArithmeticOp(*args.begin(), input), [&input](auto&& acc, auto&& expr) -> ArithmeticOp {
          return [leftArg = std::move(acc), rightArg = toArithmeticOp(std::move(expr), input)](
                     Tuple const& tuple) { return leftArg(tuple) * rightArg(tuple); };
        });
  }
  throw std::runtime_error("Unknown arithmetic operator: " + e.getHead().getName());
}

ArithmeticOp toArithmeticOp(Expression&& e, Operator const& input) {
  if(std::holds_alternative<int64_t>(e)) {
    return [val = boss::get<int64_t>(e)](Tuple const& /*tuple*/) { return val; };
  } else if(std::holds_alternative<double_t>(e)) {
    return [val = boss::get<double_t>(e)](Tuple const& /*tuple*/) { return val; };
  } else if(std::holds_alternative<Symbol>(e)) {
    return [colIndex = std::distance(input.getSchema().begin(),
                                     std::find(input.getSchema().begin(), input.getSchema().end(),
                                               boss::get<Symbol>(e).getName()))](
               Tuple const& tuple) { return tuple[colIndex]; };
  } else {
    return toArithmeticOp(boss::get<ComplexExpression>(std::move(e)), input);
  }
}

std::tuple<Schema, Projection>
toSchemaAndProjection(std::move_iterator<ExpressionArguments::iterator> asExprIt,
                      std::move_iterator<ExpressionArguments::iterator> asExprItEnd,
                      Operator const& input) {
  std::vector<ArithmeticOp> projectors;
  Schema schema;
  auto const& oldSchema = input.getSchema();
  for(; asExprIt != asExprItEnd; ++asExprIt) {
    auto asExpr = boss::get<ComplexExpression>(std::move(*asExprIt));
    auto [unused0_, unused1_, dynamics, unused2_] = std::move(asExpr).decompose();
    auto it = std::make_move_iterator(dynamics.begin());
    schema.emplace_back(boss::get<Symbol>(*it++).getName());
    projectors.emplace_back(toArithmeticOp(std::move(*it++), input));
  }
  // so far handles only column name mapping (no arithmetic expressions)
  return {std::move(schema), [projs = std::move(projectors)](Tuple const& tuple) {
            Tuple projected;
            std::transform(projs.begin(), projs.end(), std::back_inserter(projected),
                           [&tuple](auto& proj) { return proj(tuple); });
            return projected;
          }};
}

Predicate toPredicate(ComplexExpression&& e, Operator const& input) {
  if(e.getHead() == "Where"_) {
    return toPredicate(boss::get<ComplexExpression>(
                           *std::make_move_iterator(std::move(e).getDynamicArguments().begin())),
                       input);
  }
  auto [head, unused_, dynamics, unused2_] = std::move(e).decompose();
  if(head == "Greater"_) {
    auto it = std::make_move_iterator(dynamics.begin());
    return [leftArg = toArithmeticOp(std::move(*it++), input),
            rightArg = toArithmeticOp(std::move(*it++), input)](Tuple const& tuple) {
      return leftArg(tuple) > rightArg(tuple);
    };
  }
  if(head == "Equal"_) {
    auto it = std::make_move_iterator(dynamics.begin());
    return [leftArg = toArithmeticOp(std::move(*it++), input),
            rightArg = toArithmeticOp(std::move(*it++), input)](Tuple const& tuple) {
      return leftArg(tuple) == rightArg(tuple);
    };
  }
  if(head == "And"_) {
    auto it = std::make_move_iterator(dynamics.begin());
    return [leftArg = toPredicate(boss::get<ComplexExpression>(std::move(*it++)), input),
            rightArg = toPredicate(boss::get<ComplexExpression>(std::move(*it++)), input)](
               Tuple const& tuple) { return leftArg(tuple) && rightArg(tuple); };
  }
  throw std::runtime_error("Unknown predicate operator: " + head.getName());
}

} // namespace boss::engines::volcano