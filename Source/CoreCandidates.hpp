#include <ranges>
#include <Utilities.hpp>
namespace boss {


static Expression operator|(Expression&& expression, auto&& function) {
  return std::visit(utilities::overload(std::move(function), [](auto&& atom) -> Expression { return atom; }),
                    std::move(expression));
}

struct ExtractArgument {
  size_t index;
  ExtractArgument(size_t index) : index(index) {}
  Expression operator()(ComplexExpression&& complex) {
    auto [head, otherStatics, otherDynamics, spans] = std::move(complex).decompose();
    return std::move(otherDynamics.at(index));
  }
};

template <typename TargetTypeParameter> struct to {};

template <std::ranges::input_range Range, typename TargetType>
static TargetType operator|(Range range, to<TargetType> function) {
  TargetType t;
  for(auto&& v : range) {
    t.emplace_back(std::forward<decltype(v)>(v));
  }
  return std::move(t);
}
using std::views::transform;
}