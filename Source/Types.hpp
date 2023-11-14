#pragma once

#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace boss::engines::volcano {

using Value = std::variant<int64_t, double_t>;
using Tuple = std::vector<Value>;
using Schema = std::vector<std::string>;
using Predicate = std::function<bool(Tuple const&)>;
using Projection = std::function<Tuple(Tuple const&)>;
using ArithmeticOp = std::function<Value(Tuple const&)>;

Value operator+(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&rhs](auto&& lhsVal) -> Value {
        return std::visit([&lhsVal](auto&& rhsVal) -> Value { return lhsVal + rhsVal; }, rhs);
      },
      lhs);
}

Value operator*(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&rhs](auto&& lhsVal) -> Value {
        return std::visit([&lhsVal](auto&& rhsVal) -> Value { return lhsVal * rhsVal; }, rhs);
      },
      lhs);
}

bool operator>(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&rhs](auto&& lhsVal) {
        return std::visit([&lhsVal](auto&& rhsVal) { return lhsVal > rhsVal; }, rhs);
      },
      lhs);
}

bool operator<(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&rhs](auto&& lhsVal) {
        return std::visit([&lhsVal](auto&& rhsVal) { return lhsVal < rhsVal; }, rhs);
      },
      lhs);
}

bool operator==(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&rhs](auto&& lhsVal) {
        return std::visit([&lhsVal](auto&& rhsVal) { return lhsVal == rhsVal; }, rhs);
      },
      lhs);
}

} // namespace boss::engines::volcano
