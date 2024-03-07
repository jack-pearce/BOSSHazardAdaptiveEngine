#define CATCH_CONFIG_RUNNER
#include "../Source/BOSS.hpp"
#include "../Source/BootstrapEngine.hpp"
#include "../Source/ExpressionUtilities.hpp"
#include <catch2/catch.hpp>
#include <numeric>
#include <variant>

#define DEFERRED_TO_OTHER_ENGINE

using boss::Expression;
using std::string;
using std::literals::string_literals::operator""s;
using boss::utilities::operator""_;
using Catch::Generators::random;
using Catch::Generators::table;
using Catch::Generators::take;
using Catch::Generators::values;
using std::vector;
using namespace Catch::Matchers;
using boss::expressions::CloneReason;
using boss::expressions::generic::get;
using boss::expressions::generic::get_if;
using boss::expressions::generic::holds_alternative;
namespace boss {
using boss::expressions::atoms::Span;
};
using std::int32_t;
using std::int64_t;

static std::vector<string>
    librariesToTest{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// TODO: https://github.com/symbol-store/BOSS/issues/151
static boss::ComplexExpression shallowCopy(boss::ComplexExpression const& e) {
  auto const& head = e.getHead();
  auto const& dynamics = e.getDynamicArguments();
  auto const& spans = e.getSpanArguments();
  boss::ExpressionArguments dynamicsCopy;
  std::transform(dynamics.begin(), dynamics.end(), std::back_inserter(dynamicsCopy),
                 [](auto const& arg) {
                   return std::visit(
                       boss::utilities::overload(
                           [&](boss::ComplexExpression const& expr) -> boss::Expression {
                             return shallowCopy(expr);
                           },
                           [](auto const& otherTypes) -> boss::Expression { return otherTypes; }),
                       arg);
                 });
  boss::expressions::ExpressionSpanArguments spansCopy;
  std::transform(spans.begin(), spans.end(), std::back_inserter(spansCopy), [](auto const& span) {
    return std::visit(
        [](auto const& typedSpan) -> boss::expressions::ExpressionSpanArgument {
          // just do a shallow copy of the span
          // the storage's span keeps the ownership
          // (since the storage will be alive until the query finishes)
          using SpanType = std::decay_t<decltype(typedSpan)>;
          using T = std::remove_const_t<typename SpanType::element_type>;
          if constexpr(std::is_same_v<T, bool>) {
            // TODO: this would still keep const spans for bools, need to fix later
            return SpanType(typedSpan.begin(), typedSpan.size(), []() {});
          } else {
            // force non-const value for now (otherwise expressions cannot be moved)
            auto* ptr = const_cast<T*>(typedSpan.begin()); // NOLINT
            return boss::Span<T>(ptr, typedSpan.size(), []() {});
          }
        },
        span);
  });
  return boss::ComplexExpression(head, {}, std::move(dynamicsCopy), std::move(spansCopy));
}

TEST_CASE("Subspans work correctly", "[spans]") {
  auto input = boss::Span<int64_t>{std::vector<int64_t>{1, 2, 4, 3}};
  auto subrange = std::move(input).subspan(1, 3);
  CHECK(subrange.size() == 3);
  CHECK(subrange[0] == 2);
  CHECK(subrange[1] == 4);
  CHECK(subrange[2] == 3);
  auto subrange2 = boss::Span<int64_t>{std::vector<int64_t>{1, 2, 3, 2}}.subspan(2);
  CHECK(subrange2[0] == 3);
  CHECK(subrange2[1] == 2);
}

TEST_CASE("Expressions", "[expressions]") {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  using SpanArgument = boss::expressions::ExpressionSpanArgument;
  using boss::expressions::atoms::Span;
  auto const v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto const v2 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto const e = "UnevaluatedPlus"_(v1, v2);
  CHECK(e.getHead().getName() == "UnevaluatedPlus");
  CHECK(e.getArguments().at(0) == v1);
  CHECK(e.getArguments().at(1) == v2);

  SECTION("static expression arguments") {
    auto staticArgumentExpression =
        boss::expressions::ComplexExpressionWithStaticArguments<std::int64_t, std::int64_t>(
            "UnevaluatedPlus"_, {v1, v2});
    CHECK(e == staticArgumentExpression);
  }

  SECTION("span expression arguments") {
    std::array<int64_t, 2> values = {v1, v2};
    SpanArguments args;
    args.emplace_back(Span<int64_t>(&values[0], 2, nullptr));
    auto spanArgumentExpression =
        boss::expressions::ComplexExpression("UnevaluatedPlus"_, {}, {}, std::move(args));
    CHECK(e == spanArgumentExpression);
  }

  SECTION("nested span expression arguments") {
    std::array<int64_t, 2> values = {v1, v2};
    SpanArguments args;
    args.emplace_back(Span<int64_t const>(&values[0], 2, nullptr));
    auto nested = boss::expressions::ComplexExpression("UnevaluatedPlus"_, {}, {}, std::move(args));
    boss::expressions::ExpressionArguments subExpressions;
    subExpressions.push_back(std::move(nested));
    auto spanArgumentExpression =
        boss::expressions::ComplexExpression("UnevaluatedPlus"_, {}, std::move(subExpressions), {});
    CHECK("UnevaluatedPlus"_("UnevaluatedPlus"_(v1, v2)) == spanArgumentExpression);
  }
}

TEST_CASE("Expressions with static Arguments", "[expressions]") {
  SECTION("Atomic type subexpressions") {
    auto v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
    auto v2 = GENERATE(take(3, random<std::int64_t>(1, 100)));
    auto const e = boss::ComplexExpressionWithStaticArguments<std::int64_t, std::int64_t>(
        "UnevaluatedPlus"_, {v1, v2}, {}, {});
    CHECK(e.getHead().getName() == "UnevaluatedPlus");
    CHECK(e.getArguments().at(0) == v1);
    CHECK(e.getArguments().at(1) == v2);
  }
  SECTION("Complex subexpressions") {
    auto v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
    auto const e = boss::ComplexExpressionWithStaticArguments<
        boss::ComplexExpressionWithStaticArguments<std::int64_t>>(
        {"Duh"_,
         boss::ComplexExpressionWithStaticArguments<std::int64_t>{"UnevaluatedPlus"_, {v1}, {}, {}},
         {},
         {}});
    CHECK(e.getHead().getName() == "Duh");
    // TODO: this check should be enabled but requires a way to construct argument wrappers
    // from statically typed expressions
    // std::visit(
    //     [](auto&& arg) {
    //       CHECK(std::is_same_v<decltype(arg), boss::expressions::ComplexExpression>);
    //     },
    //     e.getArguments().at(0).getArgument());
  }
}

TEST_CASE("Expression Transformation", "[expressions]") {
  auto v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto v2 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto e = "UnevaluatedPlus"_(v1, v2);
  REQUIRE(*std::begin(e.getArguments()) == v1);
  get<std::int64_t>(*std::begin(e.getArguments()))++;
  REQUIRE(*std::begin(e.getArguments()) == v1 + 1);
  std::transform(std::make_move_iterator(std::begin(e.getArguments())),
                 std::make_move_iterator(std::end(e.getArguments())), e.getArguments().begin(),
                 [](auto&& e) { return get<std::int64_t>(e) + 1; });

  CHECK(e.getArguments().at(0) == v1 + 2);
  CHECK(e.getArguments().at(1) == v2 + 1);
}

TEST_CASE("Expression without arguments", "[expressions]") {
  auto const& e = "UnevaluatedPlus"_();
  CHECK(e.getHead().getName() == "UnevaluatedPlus");
}

class DummyAtom {
public:
  friend std::ostream& operator<<(std::ostream& s, DummyAtom const& /*unused*/) {
    return s << "dummy";
  }
};

TEST_CASE("Expression cast to more general expression system", "[expressions]") {
  auto a = boss::ExtensibleExpressionSystem<>::Expression("howdie"_());
  auto b = (boss::ExtensibleExpressionSystem<DummyAtom>::Expression)std::move(a);
  CHECK(
      get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b).getHead().getName() ==
      "howdie");
  auto& srcExpr = get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b);
  auto const& cexpr = std::decay_t<decltype(srcExpr)>(std::move(srcExpr));
  auto const& args = cexpr.getArguments();
  CHECK(args.empty());
}

TEST_CASE("Complex expression's argument cast to more general expression system", "[expressions]") {
  auto a = "List"_("howdie"_(1, 2, 3));
  auto const& b1 =
      (boss::ExtensibleExpressionSystem<DummyAtom>::Expression)(std::move(a).getArgument(0));
  CHECK(
      get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b1).getHead().getName() ==
      "howdie");
  auto b2 = get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b1).cloneArgument(
      1, CloneReason::FOR_TESTING);
  CHECK(get<int32_t>(b2) == 2);
}

TEST_CASE("Extract typed arguments from complex expression (using std::accumulate)",
          "[expressions]") {
  auto exprBase = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& expr0 =
      boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression(std::move(exprBase));
  auto str = [](auto const& expr) {
    auto const& args = expr.getArguments();
    return std::accumulate(
        args.begin(), args.end(), expr.getHead().getName(),
        [](auto const& accStr, auto const& arg) {
          return accStr + "_" +
                 visit(boss::utilities::overload(
                           [](auto const& value) { return std::to_string(value); },
                           [](DummyAtom const& /*value*/) { return ""s; },
                           [](boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression const&
                                  expr) { return expr.getHead().getName(); },
                           [](boss::Symbol const& symbol) { return symbol.getName(); },
                           [](std::string const& str) { return str; }),
                       arg);
        });
  }(expr0);
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Extract typed arguments from complex expression (manual iteration)", "[expressions]") {
  auto exprBase = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& expr0 =
      boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression(std::move(exprBase));
  auto str = [](auto const& expr) {
    auto const& args = expr.getArguments();
    auto size = args.size();
    auto accStr = expr.getHead().getName();
    for(int idx = 0; idx < size; ++idx) {
      accStr +=
          "_" +
          visit(boss::utilities::overload(
                    [](auto const& value) { return std::to_string(value); },
                    [](DummyAtom const& /*value*/) { return ""s; },
                    [](boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression const& expr) {
                      return expr.getHead().getName();
                    },
                    [](boss::Symbol const& symbol) { return symbol.getName(); },
                    [](std::string const& str) { return str; }),
                args.at(idx));
    }
    return accStr;
  }(expr0);
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Merge two complex expressions", "[expressions]") {
  auto delimeters = "List"_("_"_(), "_"_(), "_"_(), "_"_());
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto delimetersIt = std::make_move_iterator(delimeters.getArguments().begin());
  auto delimetersItEnd = std::make_move_iterator(delimeters.getArguments().end());
  auto exprIt = std::make_move_iterator(expr.getArguments().begin());
  auto exprItEnd = std::make_move_iterator(expr.getArguments().end());
  auto args = boss::ExpressionArguments();
  for(; delimetersIt != delimetersItEnd && exprIt != exprItEnd; ++delimetersIt, ++exprIt) {
    args.emplace_back(std::move(*delimetersIt));
    args.emplace_back(std::move(*exprIt));
  }
  auto e = boss::ComplexExpression("List"_, std::move(args));
  auto str = std::accumulate(
      e.getArguments().begin(), e.getArguments().end(), e.getHead().getName(),
      [](auto const& accStr, auto const& arg) {
        return accStr + visit(boss::utilities::overload(
                                  [](auto const& value) { return std::to_string(value); },
                                  [](boss::ComplexExpression const& expr) {
                                    return expr.getHead().getName();
                                  },
                                  [](boss::Symbol const& symbol) { return symbol.getName(); },
                                  [](std::string const& str) { return str; }),
                              arg);
      });
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Merge a static and a dynamic complex expressions", "[expressions]") {
  auto delimeters = "List"_("_"s, "_"s, "_"s, "_"s);
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto delimetersIt = std::make_move_iterator(delimeters.getArguments().begin());
  auto delimetersItEnd = std::make_move_iterator(delimeters.getArguments().end());
  auto exprIt = std::make_move_iterator(expr.getArguments().begin());
  auto exprItEnd = std::make_move_iterator(expr.getArguments().end());
  auto args = boss::ExpressionArguments();
  for(; delimetersIt != delimetersItEnd && exprIt != exprItEnd; ++delimetersIt, ++exprIt) {
    args.emplace_back(std::move(*delimetersIt));
    args.emplace_back(std::move(*exprIt));
  }
  auto e = boss::ComplexExpression("List"_, std::move(args));
  auto str = std::accumulate(
      e.getArguments().begin(), e.getArguments().end(), e.getHead().getName(),
      [](auto const& accStr, auto const& arg) {
        return accStr + visit(boss::utilities::overload(
                                  [](auto const& value) { return std::to_string(value); },
                                  [](boss::ComplexExpression const& expr) {
                                    return expr.getHead().getName();
                                  },
                                  [](boss::Symbol const& symbol) { return symbol.getName(); },
                                  [](std::string const& str) { return str; }),
                              arg);
      });
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("holds_alternative for complex expression's arguments", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  CHECK(holds_alternative<boss::ComplexExpression>(expr.getArguments().at(0)));
  CHECK(holds_alternative<int32_t>(expr.getArguments().at(1)));
  CHECK(holds_alternative<boss::Symbol>(expr.getArguments().at(2)));
  CHECK(holds_alternative<std::string>(expr.getArguments().at(3)));
}

TEST_CASE("get_if for complex expression's arguments", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& arg0 = expr.getArguments().at(0);
  auto const& arg1 = expr.getArguments().at(1);
  auto const& arg2 = expr.getArguments().at(2);
  auto const& arg3 = expr.getArguments().at(3);
  CHECK(get_if<boss::ComplexExpression>(&arg0) != nullptr);
  CHECK(get_if<int32_t>(&arg1) != nullptr);
  CHECK(get_if<boss::Symbol>(&arg2) != nullptr);
  CHECK(get_if<std::string>(&arg3) != nullptr);
  auto const& arg0args = get<boss::ComplexExpression>(arg0).getArguments();
  CHECK(arg0args.empty());
}

TEST_CASE("move expression's arguments to a new expression", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto&& movedExpr = std::move(expr);
  boss::ExpressionArguments args = movedExpr.getArguments();
  auto expr2 = boss::ComplexExpression(std::move(movedExpr.getHead()), std::move(args)); // NOLINT
  CHECK(get<boss::ComplexExpression>(expr2.getArguments().at(0)) == "howdie"_());
  CHECK(get<int32_t>(expr2.getArguments().at(1)) == 1);
  CHECK(get<boss::Symbol>(expr2.getArguments().at(2)) == "unknown"_);
  CHECK(get<std::string>(expr2.getArguments().at(3)) == "hello world"s);
}

TEST_CASE("copy expression's arguments to a new expression", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto args =
      expr.getArguments(); // TODO: this one gets the reference to the arguments
                           // when it should be a copy.
                           // Any modification/move of args will be reflected in expr's arguments!
  get<int32_t>(args.at(1)) = 2;
  auto expr2 = boss::ComplexExpression(expr.getHead(), args);
  get<int32_t>(args.at(1)) = 3;
  auto expr3 = boss::ComplexExpression(expr.getHead(), std::move(args)); // NOLINT
  // CHECK(get<int64_t>(expr.getArguments().at(1)) == 1); // fails for now (see above TODO)
  CHECK(get<int32_t>(expr2.getArguments().at(1)) == 2);
  CHECK(get<int32_t>(expr3.getArguments().at(1)) == 3);
}

TEST_CASE("copy non-const expression's arguments to ExpressionArguments", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  boss::ExpressionArguments args = expr.getArguments(); // TODO: why is it moved?
  get<int32_t>(args.at(1)) = 2;
  auto expr2 = boss::ComplexExpression(expr.getHead(), std::move(args));
  // CHECK(get<int64_t>(expr.getArguments().at(1)) == 1); // fails because args was moved (see TODO)
  CHECK(get<int32_t>(expr2.getArguments().at(1)) == 2);
}

TEST_CASE("copy const expression's arguments to ExpressionArguments)", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  boss::ExpressionArguments args = expr.getArguments();
  get<int32_t>(args.at(1)) = 2;
  auto expr2 = boss::ComplexExpression(expr.getHead(), std::move(args));
  CHECK(get<int32_t>(expr.getArguments().at(1)) == 1);
  CHECK(get<int32_t>(expr2.getArguments().at(1)) == 2);
}

TEST_CASE("move and dispatch expression's arguments", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  std::vector<boss::Symbol> symbols;
  std::vector<boss::Expression> otherExpressions;
  for(auto&& arg : (boss::ExpressionArguments)std::move(expr).getArguments()) {
    visit(boss::utilities::overload(
              [&otherExpressions](auto&& value) {
                otherExpressions.emplace_back(std::forward<decltype(value)>(value));
              },
              [&symbols](boss::Symbol&& symbol) { symbols.emplace_back(std::move(symbol)); }),
          std::move(arg));
  }
  CHECK(symbols.size() == 1);
  CHECK(symbols[0] == "unknown"_);
  CHECK(otherExpressions.size() == 3);
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with numeric Spans", "[spans]", std::int32_t, std::int64_t,
                   std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1, 1000))));
  auto v = vector<TestType>(input);
  auto s = boss::Span<TestType>(std::move(v));
  auto vectorExpression = "duh"_(std::move(s));
  REQUIRE(vectorExpression.getArguments().size() == input.size());
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(vectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with non-owning numeric Spans", "[spans]", std::int32_t,
                   std::int64_t, std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1, 1000))));
  auto v = vector<TestType>(input);
  auto s = boss::Span<TestType>(v);
  auto vectorExpression = "duh"_(std::move(s));
  REQUIRE(vectorExpression.getArguments().size() == input.size());
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(vectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with non-owning const numeric Spans", "[spans]",
                   std::int32_t, std::int64_t, std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1, 1000))));
  auto const v = vector<TestType>(input);
  auto s = boss::Span<TestType const>(v);
  auto const vectorExpression = "duh"_(std::move(s));
  REQUIRE(vectorExpression.getArguments().size() == input.size());
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(vectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Cloning Expressions with numeric Spans", "[spans][clone]", std::int32_t,
                   std::int64_t, std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1, 1000))));
  auto vectorExpression = "duh"_(boss::Span<TestType>(vector(input)));
  auto clonedVectorExpression = vectorExpression.clone(CloneReason::FOR_TESTING);
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(clonedVectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with Spans", "[spans]", std::string, boss::Symbol) {
  using std::literals::string_literals::operator""s;
  auto vals = GENERATE(take(3, chunk(5, values({"a"s, "b"s, "c"s, "d"s, "e"s, "f"s, "g"s, "h"s}))));
  auto input = vector<TestType>();
  std::transform(begin(vals), end(vals), std::back_inserter(input),
                 [](auto v) { return TestType(v); });
  auto vectorExpression = "duh"_(boss::Span<TestType>(std::move(input)));
  for(auto i = 0U; i < vals.size(); i++) {
    CHECK(vectorExpression.getArguments().at(0) == TestType(vals.at(0)));
    CHECK(vectorExpression.getArguments()[0] == TestType(vals[0]));
  }
}

TEST_CASE("Basics", "[basics]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("CatchingErrors") {
    CHECK_THROWS_MATCHES(
        engine.evaluate("EvaluateInEngines"_("List"_(9), 5)), std::bad_variant_access,
        Message("expected and actual type mismatch in expression \"9\", expected string"));
  }

  SECTION("Atomics") {
    CHECK(get<std::int32_t>(eval(boss::Expression(9))) == 9); // NOLINT
  }

  SECTION("Addition") {
    CHECK(get<std::int32_t>(eval("Plus"_(5, 4))) == 9); // NOLINT
    CHECK(get<std::int32_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_("Plus"_(2, 3), 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_("Plus"_(3, 2), 2, 2))) == 9);
  }

  SECTION("Strings") {
    CHECK(get<string>(eval("StringJoin"_((string) "howdie", (string) " ", (string) "world"))) ==
          "howdie world");
  }

  SECTION("Doubles") {
    auto const twoAndAHalf = 2.5F;
    auto const two = 2.0F;
    auto const quantum = 0.001F;
    CHECK(std::fabs(get<double>(eval("Plus"_(twoAndAHalf, twoAndAHalf))) - two * twoAndAHalf) <
          quantum);
  }

  SECTION("Booleans") {
    CHECK(get<bool>(eval("Greater"_(5, 2))));
    CHECK(!get<bool>(eval("Greater"_(2, 5))));
  }

  SECTION("Symbols") {
    CHECK(get<boss::Symbol>(eval("Symbol"_((string) "x"))).getName() == "x");
    auto expression = get<boss::ComplexExpression>(
        eval("UndefinedFunction"_(9))); // NOLINT(readability-magic-numbers)

    CHECK(expression.getHead().getName() == "UndefinedFunction");
    CHECK(get<std::int32_t>(expression.getArguments().at(0)) == 9);

    CHECK(get<std::string>(
              get<boss::ComplexExpression>(eval("UndefinedFunction"_((string) "Hello World!")))
                  .getArguments()
                  .at(0)) == "Hello World!");
  }

  SECTION("Interpolation") {
    auto thing = GENERATE(
        take(1, chunk(3, filter([](int i) { return i % 2 == 1; }, random(1, 1000))))); // NOLINT
    std::sort(begin(thing), end(thing));
    auto y = GENERATE(
        take(1, chunk(3, filter([](int i) { return i % 2 == 1; }, random(1, 1000))))); // NOLINT

    auto interpolationTable = "Table"_("Column"_("x"_, "List"_(thing[0], thing[1], thing[2])),
                                       "Column"_("y"_, "List"_(y[0], "Interpolate"_("x"_), y[2])));

    auto expectedProjectX = "Table"_("Column"_("x"_, "List"_(thing[0], thing[1], thing[2])));
    auto expectedProjectY = "Table"_("Column"_("y"_, "List"_(y[0], (y[0] + y[2]) / 2, y[2])));

    CHECK(eval("Project"_(interpolationTable.clone(CloneReason::FOR_TESTING), "As"_("x"_, "x"_))) ==
          expectedProjectX);
    CHECK(eval("Project"_(interpolationTable.clone(CloneReason::FOR_TESTING), "As"_("y"_, "y"_))) ==
          expectedProjectY);
  }

  SECTION("Relational (Ints)") {
    SECTION("Selection") {
      auto intTable = "Table"_("Column"_("Value"_, "List"_(2, 3, 1, 4, 1))); // NOLINT
      auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_("Value"_, 3))));
      CHECK(result == "Table"_("Column"_("Value"_, "List"_(4))));
    }

    SECTION("Projection") {
      auto intTable = "Table"_("Column"_("Value"_, "List"_(10, 20, 30, 40, 50))); // NOLINT

      SECTION("Plus") {
        CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                              "As"_("Result"_, "Plus"_("Value"_, "Value"_)))) ==
              "Table"_("Column"_("Result"_, "List"_(20, 40, 60, 80, 100)))); // NOLINT
      }

      SECTION("Greater") {
        CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                              "As"_("Result"_, "Greater"_("Value"_, 25)))) ==
              "Table"_("Column"_("Result"_, "List"_(false, false, true, true, true)))); // NOLINT
        CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                              "As"_("Result"_, "Greater"_(45, "Value"_)))) ==
              "Table"_("Column"_("Result"_, "List"_(true, true, true, true, false)))); // NOLINT
      }

      SECTION("Logic") {
        CHECK(eval("Project"_(
                  intTable.clone(CloneReason::FOR_TESTING),
                  "As"_("Result"_, "And"_("Greater"_("Value"_, 25), "Greater"_(45, "Value"_))))) ==
              "Table"_("Column"_("Result"_, "List"_(false, false, true, true, false)))); // NOLINT
      }
    }

    SECTION("Join") {
      auto const dataSetSize = 10;
      std::vector<int64_t> vec1(dataSetSize);
      std::vector<int64_t> vec2(dataSetSize);
      std::iota(vec1.begin(), vec1.end(), 0);
      std::iota(vec2.begin(), vec2.end(), dataSetSize);

      auto adjacency1 = "Table"_("Column"_("From"_, "List"_(boss::Span<int64_t>(vector(vec1)))),
                                 "Column"_("To"_, "List"_(boss::Span<int64_t>(vector(vec2)))));
      auto adjacency2 = "Table"_("Column"_("From2"_, "List"_(boss::Span<int64_t>(vector(vec2)))),
                                 "Column"_("To2"_, "List"_(boss::Span<int64_t>(vector(vec1)))));

      auto result = eval("Join"_(std::move(adjacency1), std::move(adjacency2),
                                 "Where"_("Equal"_("To"_, "From2"_))));

      CHECK(get<boss::ComplexExpression>(result) ==
            "Table"_("Column"_("From"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
                     "Column"_("To"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("From2"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("To2"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))));
    }

    SECTION("Join with indexes") {
      auto const dataSetSize = 10;
      std::vector<int64_t> vec1(dataSetSize);
      std::vector<int64_t> vec2(dataSetSize);
      std::iota(vec1.begin(), vec1.end(), 0);
      std::iota(vec2.begin(), vec2.end(), dataSetSize);

      auto adjacency1 = "Table"_("Column"_("From"_, "List"_(boss::Span<int64_t>(vector(vec1)))),
                                 "Column"_("To"_, "List"_(boss::Span<int64_t>(vector(vec2)))),
                                 "Index"_("From2"_, "List"_(boss::Span<int64_t>(vector(vec1)))));
      auto adjacency2 = "Table"_("Column"_("From2"_, "List"_(boss::Span<int64_t>(vector(vec2)))),
                                 "Column"_("To2"_, "List"_(boss::Span<int64_t>(vector(vec1)))),
                                 "Index"_("To"_, "List"_(boss::Span<int64_t>(vector(vec1)))));

      auto result = eval("Join"_(std::move(adjacency1), std::move(adjacency2),
                                 "Where"_("Equal"_("To"_, "From2"_))));

      CHECK(get<boss::ComplexExpression>(result) ==
            "Table"_("Column"_("From"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
                     "Column"_("To"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("From2"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("To2"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))));
    }

    SECTION("Join with indexes and multiple spans") {
      auto const dataSetSize = 10;
      std::vector<int64_t> vec1a(dataSetSize / 2);
      std::vector<int64_t> vec1b(dataSetSize / 2);
      std::vector<int64_t> vec2a(dataSetSize / 2);
      std::vector<int64_t> vec2b(dataSetSize / 2);
      std::iota(vec1a.begin(), vec1a.end(), 0);
      std::iota(vec1b.begin(), vec1b.end(), dataSetSize / 2);
      std::iota(vec2a.begin(), vec2a.end(), dataSetSize);
      std::iota(vec2b.begin(), vec2b.end(), dataSetSize * 3 / 2);

      auto adjacency1 = "Table"_("Column"_("From"_, "List"_(boss::Span<int64_t>(vector(vec1a)),
                                                            boss::Span<int64_t>(vector(vec1b)))),
                                 "Column"_("To"_, "List"_(boss::Span<int64_t>(vector(vec2a)),
                                                          boss::Span<int64_t>(vector(vec2b)))),
                                 "Index"_("From2"_, "List"_(boss::Span<int64_t>(vector(vec1a)),
                                                            boss::Span<int64_t>(vector(vec1b)))));
      INFO(adjacency1);
      auto adjacency2 = "Table"_("Column"_("From2"_, "List"_(boss::Span<int64_t>(vector(vec2a)),
                                                             boss::Span<int64_t>(vector(vec2b)))),
                                 "Column"_("To2"_, "List"_(boss::Span<int64_t>(vector(vec1a)),
                                                           boss::Span<int64_t>(vector(vec1b)))),
                                 "Index"_("To"_, "List"_(boss::Span<int64_t>(vector(vec1a)),
                                                         boss::Span<int64_t>(vector(vec1b)))));
      INFO(adjacency2);

      auto result = eval("Join"_(std::move(adjacency1), std::move(adjacency2),
                                 "Where"_("Equal"_("To"_, "From2"_))));

      CHECK(get<boss::ComplexExpression>(result) ==
            "Table"_("Column"_("From"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
                     "Column"_("To"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("From2"_, "List"_(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
                     "Column"_("To2"_, "List"_(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))));
    }
  }

  SECTION("Relational (Strings)") {
    auto customerTable = "Table"_("Column"_("FirstName"_, "List"_("John", "Sam", "Barbara")),
                                  "Column"_("LastName"_, "List"_("McCarthy", "Madden", "Liskov")));

    SECTION("Selection") {
      auto sam = eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                "Where"_("StringContainsQ"_("LastName"_, "Madden"))));
      CHECK(sam == "Table"_("Column"_("FirstName"_, "List"_("Sam")),
                            "Column"_("LastName"_, "List"_("Madden"))));
    }

    SECTION("Aggregation") {
      SECTION("ConstantGroup") {
        auto result =
            eval("Group"_(customerTable.clone(CloneReason::FOR_TESTING), "Function"_(0), "Count"_));
        INFO(result);
        CHECK(get<boss::ComplexExpression>(result).getArguments().size() == 2);
        CHECK(get<boss::ComplexExpression>(
                  get<boss::ComplexExpression>(
                      get<boss::ComplexExpression>(result).getArguments().at(0))
                      .getArguments()
                      .at(1)) == "List"_(0));
        CHECK(get<boss::ComplexExpression>(
                  get<boss::ComplexExpression>(
                      get<boss::ComplexExpression>(result).getArguments().at(1))
                      .getArguments()
                      .at(1)) == "List"_(3));
      }

      SECTION("NoGroup") {
        auto result = eval("Group"_(customerTable.clone(CloneReason::FOR_TESTING), "Count"_));
        INFO(result);
        CHECK(get<boss::ComplexExpression>(result).getArguments().size() == 1);
        CHECK(get<boss::ComplexExpression>(
                  get<boss::ComplexExpression>(
                      get<boss::ComplexExpression>(result).getArguments().at(0))
                      .getArguments()
                      .at(1)) == "List"_(3));
      }

      SECTION("Select+Group") {
        auto result = eval("Group"_("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                              "Where"_("StringContainsQ"_("LastName"_, "Madden"))),
                                    "Function"_(0), "Count"_));
        INFO(result);
        CHECK(get<boss::ComplexExpression>(result).getArguments().size() == 2);
        CHECK(get<boss::ComplexExpression>(
                  get<boss::ComplexExpression>(
                      get<boss::ComplexExpression>(result).getArguments().at(0))
                      .getArguments()
                      .at(1)) == "List"_(0));
        CHECK(get<boss::ComplexExpression>(
                  get<boss::ComplexExpression>(
                      get<boss::ComplexExpression>(result).getArguments().at(1))
                      .getArguments()
                      .at(1)) == "List"_(1));
      }
    }
  }

  SECTION("Relational (empty table)") {
    auto emptyCustomerTable =
        "Table"_("Column"_("ID"_, "List"_()), "Column"_("FirstName"_, "List"_()),
                 "Column"_("LastName"_, "List"_()), "Column"_("BirthYear"_, "List"_()),
                 "Column"_("Country"_, "List"_()));
    auto emptySelect =
        eval("Select"_(emptyCustomerTable.clone(CloneReason::FOR_TESTING), "Function"_(true)));
    CHECK(emptySelect == emptyCustomerTable);
  }

  SECTION("Relational (multiple types)") {
    auto customerTable = "Table"_("Column"_("ID"_, "List"_(1, 2, 3)), // NOLINT
                                  "Column"_("FirstName"_, "List"_("John", "Sam", "Barbara")),
                                  "Column"_("LastName"_, "List"_("McCarthy", "Madden", "Liskov")),
                                  "Column"_("BirthYear"_, "List"_(1927, 1976, 1939)), // NOLINT
                                  "Column"_("Country"_, "List"_("USA", "USA", "USA")));

    SECTION("Selection") {
      auto fullTable =
          eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING), "Function"_(true)));
      CHECK(fullTable == customerTable);

      auto none =
          eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING), "Function"_(false)));
      CHECK(none == "Table"_("Column"_("ID"_, "List"_()), "Column"_("FirstName"_, "List"_()),
                             "Column"_("LastName"_, "List"_()), "Column"_("BirthYear"_, "List"_()),
                             "Column"_("Country"_, "List"_())));

      auto usa = eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                "Where"_("StringContainsQ"_("Country"_, "USA"))));
      CHECK(usa == customerTable);

      auto madden = eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                   "Where"_("StringContainsQ"_("LastName"_, "Madden"))));
      CHECK(madden == "Table"_("Column"_("ID"_, "List"_(2)), // NOLINT
                               "Column"_("FirstName"_, "List"_("Sam")),
                               "Column"_("LastName"_, "List"_("Madden")),
                               "Column"_("BirthYear"_, "List"_(1976)), // NOLINT
                               "Column"_("Country"_, "List"_("USA"))));

      auto john = eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                 "Where"_("StringContainsQ"_("FirstName"_, "John"))));
      CHECK(john == "Table"_("Column"_("ID"_, "List"_(1)), // NOLINT
                             "Column"_("FirstName"_, "List"_("John")),
                             "Column"_("LastName"_, "List"_("McCarthy")),
                             "Column"_("BirthYear"_, "List"_(1927)), // NOLINT
                             "Column"_("Country"_, "List"_("USA"))));

      auto id3 = eval(
          "Select"_(customerTable.clone(CloneReason::FOR_TESTING), "Where"_("Equal"_("ID"_, 3))));
      CHECK(id3 == "Table"_("Column"_("ID"_, "List"_(3)), // NOLINT
                            "Column"_("FirstName"_, "List"_("Barbara")),
                            "Column"_("LastName"_, "List"_("Liskov")),
                            "Column"_("BirthYear"_, "List"_(1939)), // NOLINT
                            "Column"_("Country"_, "List"_("USA"))));

      auto notFound = eval("Select"_(customerTable.clone(CloneReason::FOR_TESTING),
                                     "Where"_("Equal"_("BirthYear"_, 0))));
      CHECK(notFound == "Table"_("Column"_("ID"_, "List"_()), "Column"_("FirstName"_, "List"_()),
                                 "Column"_("LastName"_, "List"_()),
                                 "Column"_("BirthYear"_, "List"_()),
                                 "Column"_("Country"_, "List"_())));
    }

    SECTION("Projection") {
      auto fullnames =
          eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                          "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
      CHECK(fullnames == "Table"_("Column"_("FirstName"_, "List"_("John", "Sam", "Barbara")),
                                  "Column"_("LastName"_, "List"_("McCarthy", "Madden", "Liskov"))));
      auto firstNames = eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                                        "As"_("FirstName"_, "FirstName"_)));
      CHECK(firstNames == "Table"_("Column"_("FirstName"_, "List"_("John", "Sam", "Barbara"))));
      auto lastNames = eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                                       "As"_("LastName"_, "LastName"_)));
      CHECK(lastNames == "Table"_("Column"_("LastName"_, "List"_("McCarthy", "Madden", "Liskov"))));
    }

    SECTION("Sorting") {
      auto sortedByID =
          eval("Sort"_("Select"_(customerTable.clone(CloneReason::FOR_TESTING), "Function"_(true)),
                       "By"_("ID"_)));
      CHECK(sortedByID == customerTable);

      auto sortedByLastName =
          eval("Sort"_("Select"_(customerTable.clone(CloneReason::FOR_TESTING), "Function"_(true)),
                       "By"_("LastName"_)));
      CHECK(sortedByLastName ==
            "Table"_("Column"_("ID"_, "List"_(3, 2, 1)), // NOLINT
                     "Column"_("FirstName"_, "List"_("Barbara", "Sam", "John")),
                     "Column"_("LastName"_, "List"_("Liskov", "Madden", "McCarthy")),
                     "Column"_("BirthYear"_, "List"_(1939, 1976, 1927)), // NOLINT
                     "Column"_("Country"_, "List"_("USA", "USA", "USA"))));
    }

    SECTION("Aggregation") {
      auto countRows = eval("Group"_("Customer"_, "Function"_(0), "Count"_));
      INFO(countRows);
      CHECK(get<boss::ComplexExpression>(countRows).getArguments().size() == 2);
      CHECK(get<boss::ComplexExpression>(
                get<boss::ComplexExpression>(
                    get<boss::ComplexExpression>(countRows).getArguments().at(0))
                    .getArguments()
                    .at(1)) == "List"_(0));
      CHECK(get<boss::ComplexExpression>(
                get<boss::ComplexExpression>(
                    get<boss::ComplexExpression>(countRows).getArguments().at(1))
                    .getArguments()
                    .at(1)) == "List"_(3));
    }
  }
}

static int64_t operator""_i64(char c) { return static_cast<int64_t>(c); };

TEST_CASE("TPC-H", "[tpch]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto multipleSpans = GENERATE(false, true);

  auto asInt64Spans = [&eval, &multipleSpans](auto&& val0, auto&&... val) {
    auto evalIfNeeded = [&eval](auto&& val) {
      if constexpr(std::is_same_v<boss::ComplexExpression, std::decay_t<decltype(val)>>) {
        return get<int64_t>(eval(std::move(val)));
      } else {
        return (int64_t)std::move(val);
      }
    };
    boss::expressions::ExpressionSpanArguments spans;
    if(multipleSpans) {
      spans.emplace_back(boss::Span<int64_t>{std::vector<int64_t>{evalIfNeeded(std::move(val0))}});
      spans.emplace_back(
          boss::Span<int64_t>{std::vector<int64_t>{evalIfNeeded(std::move(val))...}});
    } else {
      spans.emplace_back(boss::Span<int64_t>{
          std::vector<int64_t>{evalIfNeeded(std::move(val0)), evalIfNeeded(std::move(val))...}});
    }
    return boss::ComplexExpression("List"_, {}, {}, std::move(spans));
  };

  auto asDoubleSpans = [&eval, &multipleSpans](auto&& val0, auto&&... val) {
    auto evalIfNeeded = [&eval](auto&& val) {
      if constexpr(std::is_same_v<boss::ComplexExpression, std::decay_t<decltype(val)>>) {
        return get<double_t>(eval(std::move(val)));
      } else {
        return (double_t)std::move(val);
      }
    };
    boss::expressions::ExpressionSpanArguments spans;
    if(multipleSpans) {
      spans.emplace_back(
          boss::Span<double_t>{std::vector<double_t>{evalIfNeeded(std::move(val0))}});
      spans.emplace_back(
          boss::Span<double_t>{std::vector<double_t>{evalIfNeeded(std::move(val))...}});
    } else {
      spans.emplace_back(boss::Span<double_t>{
          std::vector<double_t>{evalIfNeeded(std::move(val0)), evalIfNeeded(std::move(val))...}});
    }
    return boss::ComplexExpression("List"_, {}, {}, std::move(spans));
  };

  auto nation = "Table"_(
      "Column"_("N_NATIONKEY"_, asInt64Spans(1, 2, 3, 4)), // NOLINT
      "Column"_("N_REGIONKEY"_, asInt64Spans(1, 1, 2, 3)), // NOLINT
      "Column"_("N_NAME"_, "DictionaryEncodedList"_(asInt64Spans(0, 7, 16, 22, 28), "ALGERIA"
                                                                                    "ARGENTINA"
                                                                                    "BRAZIL"
                                                                                    "CANADA")),
      "Index"_("R_REGIONKEY"_, asInt64Spans(1, 1, 2, 3)));

  auto part = "Table"_(
      "Column"_("P_PARTKEY"_, asInt64Spans(4, 3, 2, 1)),                          // NOLINT
      "Column"_("P_RETAILPRICE"_, asDoubleSpans(100.01, 100.01, 100.01, 100.01)), // NOLINT
      "Column"_("P_NAME"_, "DictionaryEncodedList"_(asInt64Spans(0, 35, 72, 107, 144),
                                                    "spring green yellow purple cornsilk"
                                                    "cornflower chocolate smoke green pink"
                                                    "moccasin green thistle khaki floral"
                                                    "green blush tomato burlywood seashell")));

  auto supplier = "Table"_("Column"_("S_SUPPKEY"_, asInt64Spans(1, 4, 2, 3)),   // NOLINT
                           "Column"_("S_NATIONKEY"_, asInt64Spans(1, 1, 2, 3)), // NOLINT
                           "Index"_("N_NATIONKEY"_, asInt64Spans(0, 0, 1, 2))); // NOLINT

  auto partsupp =
      "Table"_("Column"_("PS_PARTKEY"_, asInt64Spans(1, 2, 3, 4)),                         // NOLINT
               "Column"_("PS_SUPPKEY"_, asInt64Spans(1, 2, 3, 4)),                         // NOLINT
               "Column"_("PS_SUPPLYCOST"_, asDoubleSpans(771.64, 993.49, 337.09, 357.84)), // NOLINT
               "Index"_("P_PARTKEY"_, asInt64Spans(3, 3, 2, 1)),                           // NOLINT
               "Index"_("S_SUPPKEY"_, asInt64Spans(0, 0, 2, 3)));                          // NOLINT

  auto customer =
      "Table"_("Column"_("C_CUSTKEY"_, asInt64Spans(4, 7, 1, 4)),                        // NOLINT
               "Column"_("C_NATIONKEY"_, asInt64Spans(3, 3, 1, 4)),                      // NOLINT
               "Column"_("C_ACCTBAL"_, asDoubleSpans(711.56, 121.65, 7498.12, 2866.83)), // NOLINT
               "Column"_("C_NAME"_, "DictionaryEncodedList"_(asInt64Spans(0, 18, 36, 54, 72),
                                                             "Customer#000000001"
                                                             "Customer#000000002"
                                                             "Customer#000000003"
                                                             "Customer#000000004")),
               "Column"_("C_MKTSEGMENT"_,
                         "DictionaryEncodedList"_(asInt64Spans(0, 10, 19, 28, 36), "AUTOMOBILE"
                                                                                   "MACHINERY"
                                                                                   "HOUSEHOLD"
                                                                                   "BUILDING")),
               "Index"_("N_NATIONKEY"_, asInt64Spans(2, 2, 0, 3))); // NOLINT

  auto orders =
      "Table"_("Column"_("O_ORDERKEY"_, asInt64Spans(1, 0, 2, 3)),
               "Column"_("O_CUSTKEY"_, asInt64Spans(4, 7, 1, 4)), // NOLINT
               "Column"_("O_TOTALPRICE"_,
                         asDoubleSpans(178821.73, 154260.84, 202660.52, 155680.60)), // NOLINT
               "Column"_("O_ORDERDATE"_,
                         asInt64Spans("DateObject"_("1998-01-24"), "DateObject"_("1992-05-01"),
                                      "DateObject"_("1992-12-21"), "DateObject"_("1994-06-18"))),
               "Column"_("O_SHIPPRIORITY"_, asInt64Spans(1, 1, 1, 1)), // NOLINT
               "Index"_("C_CUSTKEY"_, asInt64Spans(0, 1, 2, 3)));      // NOLINT

  auto lineitem = "Table"_(
      "Column"_("L_ORDERKEY"_, asInt64Spans(1, 1, 2, 3)), // NOLINT
      "Column"_("L_PARTKEY"_, asInt64Spans(1, 2, 3, 4)),  // NOLINT
      "Column"_("L_SUPPKEY"_, asInt64Spans(1, 2, 3, 4)),  // NOLINT
      "Column"_("L_RETURNFLAG"_,
                "DictionaryEncodedList"_(asInt64Spans(0, 1, 2, 3), "NNAA")), // NOLINT
      "Column"_("L_LINESTATUS"_,
                "DictionaryEncodedList"_(asInt64Spans(0, 1, 2, 3), "OOFF")),               // NOLINT
      "Column"_("L_RETURNFLAG_INT"_, asInt64Spans('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),    // NOLINT
      "Column"_("L_LINESTATUS_INT"_, asInt64Spans('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),    // NOLINT
      "Column"_("L_QUANTITY"_, asInt64Spans(17, 21, 8, 5)),                                // NOLINT
      "Column"_("L_EXTENDEDPRICE"_, asDoubleSpans(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
      "Column"_("L_DISCOUNT"_, asDoubleSpans(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
      "Column"_("L_TAX"_, asDoubleSpans(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
      "Column"_("L_SHIPDATE"_,
                asInt64Spans("DateObject"_("1992-03-13"), "DateObject"_("1994-04-12"),
                             "DateObject"_("1996-02-28"), "DateObject"_("1994-12-31"))),
      "Index"_("O_ORDERKEY"_, asInt64Spans(0, 0, 2, 3)),            // NOLINT
      "Index"_("PS_PARTKEYPS_SUPPKEY"_, asInt64Spans(0, 1, 2, 3))); // NOLINT

  auto useCache = GENERATE(false, true);

  if(useCache) {
    CHECK(eval("Set"_("CachedColumn"_, "L_QUANTITY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_DISCOUNT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_SHIPDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_EXTENDEDPRICE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_QUANTITY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_DISCOUNT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_SHIPDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_EXTENDEDPRICE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_RETURNFLAG"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_LINESTATUS"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_TAX"_)) == true);
    // Q3
    CHECK(eval("Set"_("CachedColumn"_, "C_CUSTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "C_MKTSEGMENT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_CUSTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_SHIPPRIORITY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_DISCOUNT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_SHIPDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_EXTENDEDPRICE"_)) == true);
    // Q6
    CHECK(eval("Set"_("CachedColumn"_, "L_QUANTITY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_DISCOUNT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_SHIPDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_EXTENDEDPRICE"_)) == true);
    // Q9
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "P_PARTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "P_RETAILPRICE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "N_NAME"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "N_NATIONKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "S_SUPPKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "S_NATIONKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "PS_PARTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "PS_SUPPKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "PS_SUPPLYCOST"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_PARTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_SUPPKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_EXTENDEDPRICE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_DISCOUNT"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_QUANTITY"_)) == true);
    // Q18
    CHECK(eval("Set"_("CachedColumn"_, "L_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "L_QUANTITY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "C_CUSTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_CUSTKEY"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_ORDERDATE"_)) == true);
    CHECK(eval("Set"_("CachedColumn"_, "O_TOTALPRICE"_)) == true);
  }

  auto const& [queryName, query,
               expectedOutput] = GENERATE_REF(table<std::string,
                                                    std::function<boss::ComplexExpression(void)>,
                                                    std::function<boss::Expression(void)>>(
      {{"Q1 (Select only)",
        [&]() {
          return "Select"_("Project"_(shallowCopy(lineitem), "As"_("L_SHIPDATE"_, "L_SHIPDATE"_)),
                           "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_)));
        },
        [&]() {
          return eval("Table"_("Column"_(
              "L_SHIPDATE"_, "List"_("DateObject"_("1992-03-13"), "DateObject"_("1994-04-12"),
                                     "DateObject"_("1996-02-28"), "DateObject"_("1994-12-31")))));
        }},
       {"Q1 (Project only)",
        [&]() {
          return "Project"_(
              "Project"_(shallowCopy(lineitem),
                         "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                               "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                               "Plus"_("L_TAX"_, 1.0), "L_DISCOUNT"_, "L_DISCOUNT"_)),
              "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                    "disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                    "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                    "L_DISCOUNT"_));
        },
        []() {
          return "Table"_(
              "Column"_("L_QUANTITY"_, "List"_(17, 21, 8, 5)), // NOLINT
              "Column"_("L_EXTENDEDPRICE"_,
                        "List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
              "Column"_("disc_price"_,
                        "List"_(17954.55 * (1.0 - 0.10), 34850.16 * (1.0 - 0.05),    // NOLINT
                                7712.48 * (1.0 - 0.06), 25284.00 * (1.0 - 0.06))),   // NOLINT
              "Column"_("charge"_, "List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0),   // NOLINT
                                           34850.16 * (1.0 - 0.05) * (0.06 + 1.0),   // NOLINT
                                           7712.48 * (1.0 - 0.06) * (0.02 + 1.0),    // NOLINT
                                           25284.00 * (1.0 - 0.06) * (0.06 + 1.0))), // NOLINT
              "Column"_("L_DISCOUNT"_, "List"_(0.10, 0.05, 0.06, 0.06)));            // NOLINT
        }},
       {"Q1 (Select-Project only)",
        [&]() {
          return "Project"_(
              "Project"_(
                  "Project"_(
                      "Select"_("Project"_(shallowCopy(lineitem),
                                           "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                 "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                 "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_TAX"_,
                                                 "L_TAX"_)),
                                "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                      "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                            "L_DISCOUNT"_, "L_DISCOUNT"_, "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_),
                            "calc2"_, "Plus"_("L_TAX"_, 1.0))),
                  "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                        "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_,
                        "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc2"_, "calc2"_)),
              "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                    "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_, "disc_price"_, "calc"_,
                    "Times"_("disc_price"_, "calc2"_)));
        },
        []() {
          return "Table"_(
              "Column"_("L_QUANTITY"_, "List"_(17, 21, 8, 5)), // NOLINT
              "Column"_("L_EXTENDEDPRICE"_,
                        "List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
              "Column"_("L_DISCOUNT"_, "List"_(0.10, 0.05, 0.06, 0.06)), // NOLINT
              "Column"_("disc_price"_,
                        "List"_(17954.55 * (1.0 - 0.10), 34850.16 * (1.0 - 0.05),   // NOLINT
                                7712.48 * (1.0 - 0.06), 25284.00 * (1.0 - 0.06))),  // NOLINT
              "Column"_("calc"_, "List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0),    // NOLINT
                                         34850.16 * (1.0 - 0.05) * (0.06 + 1.0),    // NOLINT
                                         7712.48 * (1.0 - 0.06) * (0.02 + 1.0),     // NOLINT
                                         25284.00 * (1.0 - 0.06) * (0.06 + 1.0)))); // NOLINT
        }},
       {"Q1 (No Order, No Strings)",
        [&]() {
          return "Group"_(
              "Project"_(
                  "Project"_(
                      "Project"_(
                          "Select"_(
                              "Project"_(shallowCopy(lineitem),
                                         "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                               "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                               "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                               "L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_,
                                               "L_LINESTATUS_INT"_, "L_LINESTATUS_INT"_, "L_TAX"_,
                                               "L_TAX"_)),
                              "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                          "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
                                "L_LINESTATUS_INT"_, "L_QUANTITY"_, "L_QUANTITY"_,
                                "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                "L_DISCOUNT"_, "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                                "Plus"_("L_TAX"_, 1.0))),
                      "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
                            "L_LINESTATUS_INT"_, "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                            "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_,
                            "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc2"_, "calc2"_)),
                  "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
                        "L_LINESTATUS_INT"_, "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                        "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_,
                        "disc_price"_, "calc"_, "Times"_("disc_price"_, "calc2"_))),
              "By"_("L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_),
              "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                    "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("disc_price"_),
                    "SUM_CHARGES"_, "Sum"_("calc"_), "AVG_QTY"_, "Avg"_("L_QUANTITY"_),
                    "AVG_PRICE"_, "Avg"_("L_EXTENDEDPRICE"_), "AVG_DISC"_, "Avg"_("l_discount"_),
                    "COUNT_ORDER"_, "Count"_("*"_)));
        },
        []() {
          return "Table"_(
              "Column"_("L_RETURNFLAG_INT"_, "List"_('N'_i64, 'A'_i64)), // NOLINT
              "Column"_("L_LINESTATUS_INT"_, "List"_('O'_i64, 'F'_i64)), // NOLINT
              "Column"_("SUM_QTY"_, "List"_(17 + 21, 8 + 5)),            // NOLINT
              "Column"_("SUM_BASE_PRICE"_,
                        "List"_(17954.55 + 34850.16, 7712.48 + 25284.00)), // NOLINT
              "Column"_("SUM_DISC_PRICE"_,
                        "List"_(17954.55 * (1.0 - 0.10) + 34850.16 * (1.0 - 0.05),  // NOLINT
                                7712.48 * (1.0 - 0.06) + 25284.00 * (1.0 - 0.06))), // NOLINT
              "Column"_("SUM_CHARGES"_,
                        "List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0) +             // NOLINT
                                    34850.16 * (1.0 - 0.05) * (0.06 + 1.0),          // NOLINT
                                7712.48 * (1.0 - 0.06) * (0.02 + 1.0) +              // NOLINT
                                    25284.00 * (1.0 - 0.06) * (0.06 + 1.0))),        // NOLINT
              "Column"_("AVG_PRICE"_, "List"_((17954.55 + 34850.16) / 2,             // NOLINT
                                              (7712.48 + 25284.00) / 2)),            // NOLINT
              "Column"_("AVG_DISC"_, "List"_((0.10 + 0.05) / 2, (0.06 + 0.06) / 2)), // NOLINT
              "Column"_("COUNT_ORDER"_, "List"_(2, 2)));                             // NOLINT
        }},
       {"Q1 (No Order)",
        [&]() {
          return "Group"_(
              "Project"_(
                  "Project"_(
                      "Project"_(
                          "Select"_(
                              "Project"_(shallowCopy(lineitem),
                                         "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                               "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                               "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                               "L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_,
                                               "L_LINESTATUS"_, "L_TAX"_, "L_TAX"_)),
                              "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                          "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                                "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                                "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "calc1"_,
                                "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_("L_TAX"_, 1.0))),
                      "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                            "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                            "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_,
                            "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc2"_, "calc2"_)),
                  "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                        "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                        "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_, "disc_price"_, "calc"_,
                        "Times"_("disc_price"_, "calc2"_))),
              "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_),
              "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                    "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("disc_price"_),
                    "SUM_CHARGES"_, "Sum"_("calc"_), "AVG_QTY"_, "Avg"_("L_QUANTITY"_),
                    "AVG_PRICE"_, "Avg"_("L_EXTENDEDPRICE"_), "AVG_DISC"_, "Avg"_("l_discount"_),
                    "COUNT_ORDER"_, "Count"_("*"_)));
        },
        []() {
          return "Table"_(
              "Column"_("L_RETURNFLAG"_, "List"_("N", "A")),  // NOLINT
              "Column"_("L_LINESTATUS"_, "List"_("O", "F")),  // NOLINT
              "Column"_("SUM_QTY"_, "List"_(17 + 21, 8 + 5)), // NOLINT
              "Column"_("SUM_BASE_PRICE"_,
                        "List"_(17954.55 + 34850.16, 7712.48 + 25284.00)), // NOLINT
              "Column"_("SUM_DISC_PRICE"_,
                        "List"_(17954.55 * (1.0 - 0.10) + 34850.16 * (1.0 - 0.05),  // NOLINT
                                7712.48 * (1.0 - 0.06) + 25284.00 * (1.0 - 0.06))), // NOLINT
              "Column"_("SUM_CHARGES"_,
                        "List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0) +             // NOLINT
                                    34850.16 * (1.0 - 0.05) * (0.06 + 1.0),          // NOLINT
                                7712.48 * (1.0 - 0.06) * (0.02 + 1.0) +              // NOLINT
                                    25284.00 * (1.0 - 0.06) * (0.06 + 1.0))),        // NOLINT
              "Column"_("AVG_PRICE"_, "List"_((17954.55 + 34850.16) / 2,             // NOLINT
                                              (7712.48 + 25284.00) / 2)),            // NOLINT
              "Column"_("AVG_DISC"_, "List"_((0.10 + 0.05) / 2, (0.06 + 0.06) / 2)), // NOLINT
              "Column"_("COUNT_ORDER"_, "List"_(2, 2)));                             // NOLINT
        }},
       {"Q1",
        [&]() {
          return "Order"_(
              "Group"_(
                  "Project"_(
                      "Project"_(
                          "Project"_(
                              "Select"_(
                                  "Project"_(shallowCopy(lineitem),
                                             "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                   "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                   "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                   "L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                   "L_LINESTATUS"_, "L_LINESTATUS"_, "L_TAX"_,
                                                   "L_TAX"_)),
                                  "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                              "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_,
                                    "L_LINESTATUS"_, "L_QUANTITY"_, "L_QUANTITY"_,
                                    "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                    "L_DISCOUNT"_, "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                                    "Plus"_("L_TAX"_, 1.0))),
                          "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                                "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                                "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_,
                                "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc2"_, "calc2"_)),
                      "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                            "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                            "L_DISCOUNT"_, "L_DISCOUNT"_, "disc_price"_, "disc_price"_, "calc"_,
                            "Times"_("disc_price"_, "calc2"_))),
                  "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_),
                  "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                        "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("disc_price"_),
                        "SUM_CHARGES"_, "Sum"_("calc"_), "AVG_QTY"_, "Avg"_("L_QUANTITY"_),
                        "AVG_PRICE"_, "Avg"_("L_EXTENDEDPRICE"_), "AVG_DISC"_,
                        "Avg"_("l_discount"_), "COUNT_ORDER"_, "Count"_("*"_))),
              "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_));
        },
        []() {
          return "Table"_(
              "Column"_("L_RETURNFLAG"_, "List"_("A", "N")),  // NOLINT
              "Column"_("L_LINESTATUS"_, "List"_("F", "O")),  // NOLINT
              "Column"_("SUM_QTY"_, "List"_(8 + 5, 17 + 21)), // NOLINT
              "Column"_("SUM_BASE_PRICE"_,
                        "List"_(7712.48 + 25284.00, 17954.55 + 34850.16)), // NOLINT
              "Column"_("SUM_DISC_PRICE"_,
                        "List"_(7712.48 * (1.0 - 0.06) + 25284.00 * (1.0 - 0.06),    // NOLINT
                                17954.55 * (1.0 - 0.10) + 34850.16 * (1.0 - 0.05))), // NOLINT
              "Column"_("SUM_CHARGES"_,
                        "List"_(7712.48 * (1.0 - 0.06) * (0.02 + 1.0) +              // NOLINT
                                    25284.00 * (1.0 - 0.06) * (0.06 + 1.0),          // NOLINT
                                17954.55 * (1.0 - 0.10) * (0.02 + 1.0) +             // NOLINT
                                    34850.16 * (1.0 - 0.05) * (0.06 + 1.0))),        // NOLINT
              "Column"_("AVG_PRICE"_, "List"_((7712.48 + 25284.00) / 2,              // NOLINT
                                              (17954.55 + 34850.16) / 2)),           // NOLINT
              "Column"_("AVG_DISC"_, "List"_((0.06 + 0.06) / 2, (0.10 + 0.05) / 2)), // NOLINT
              "Column"_("COUNT_ORDER"_, "List"_(2, 2)));                             // NOLINT
        }},
       {"Q6 (No Grouping)",
        [&]() {
          return "Project"_(
              "Select"_("Project"_(shallowCopy(lineitem),
                                   "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                         "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_,
                                         "L_EXTENDEDPRICE"_)),
                        "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),      // NOLINT
                                        "Greater"_("L_DISCOUNT"_, 0.0499),  // NOLINT
                                        "Greater"_(0.07001, "L_DISCOUNT"_), // NOLINT
                                        "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                        "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
              "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_)));
        },
        []() {
          return "Table"_(
              "Column"_("revenue"_, "List"_(34850.16 * 0.05, 25284.00 * 0.06))); // NOLINT
        }},
       {"Q6",
        [&]() {
          return "Group"_(
              "Project"_("Select"_("Project"_(shallowCopy(lineitem),
                                              "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                    "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                    "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                   "Where"_("And"_(
                                       "Greater"_(24, "L_QUANTITY"_),      // NOLINT
                                       "Greater"_("L_DISCOUNT"_, 0.0499),  // NOLINT
                                       "Greater"_(0.07001, "L_DISCOUNT"_), // NOLINT
                                       "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                       "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
                         "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
              "Sum"_("revenue"_));
        },
        []() {
          return "Table"_(
              "Column"_("revenue"_, "List"_(34850.16 * 0.05 + 25284.00 * 0.06))); // NOLINT
        }},
       {"Q6 (AF Heuristics)",
        [&]() {
          return "Group"_(
              "Project"_(
                  "Select"_(
                      "Select"_(
                          "Select"_("Project"_(shallowCopy(lineitem),
                                               "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                     "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                     "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                    "Where"_("Greater"_(24, "L_QUANTITY"_))),    // NOLINT
                          "Where"_("And"_("Greater"_("L_DISCOUNT"_, 0.0499),     // NOLINT
                                          "Greater"_(0.07001, "L_DISCOUNT"_)))), // NOLINT
                      "Where"_("And"_("Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                      "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
                  "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
              "Sum"_("revenue"_));
        },
        []() {
          return "Table"_(
              "Column"_("revenue"_, "List"_(34850.16 * 0.05 + 25284.00 * 0.06))); // NOLINT
        }},
       {"Q3 (No Strings)",
        [&]() {
          return "Top"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Project"_(
                              "Join"_(
                                  "Project"_(
                                      "Select"_(
                                          "Project"_(shallowCopy(customer),
                                                     "As"_("C_CUSTKEY"_, "C_CUSTKEY"_, "C_ACCTBAL"_,
                                                           "C_ACCTBAL"_)),
                                          "Where"_("Equal"_("C_ACCTBAL"_, 2866.83))), // NOLINT
                                      "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
                                  "Select"_(
                                      "Project"_(shallowCopy(orders),
                                                 "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                       "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                       "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                                      "Where"_(
                                          "Greater"_("DateObject"_("1995-03-15"), "O_ORDERDATE"_))),
                                  "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                              "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                    "O_CUSTKEY"_, "O_CUSTKEY"_, "O_SHIPPRIORITY"_,
                                    "O_SHIPPRIORITY"_)),
                          "Project"_(
                              "Select"_(
                                  "Project"_(shallowCopy(lineitem),
                                             "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_,
                                                   "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                   "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                  "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1993-03-15")))),
                              "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                    "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                          "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                      "As"_("Expr1009"_,
                            "Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)), // NOLINT
                            "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                            "O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                  "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_),
                  "As"_("revenue"_, "Sum"_("Expr1009"_))),
              "By"_("revenue"_, "desc"_, "O_ORDERDATE"_), 10); // NOLINT
        },
        []() { return "Dummy"_(); }},
       {"Q3",
        [&]() {
          return "Top"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Project"_(
                              "Join"_(
                                  "Project"_(
                                      "Select"_("Project"_(shallowCopy(customer),
                                                           "As"_("C_CUSTKEY"_, "C_CUSTKEY"_,
                                                                 "C_MKTSEGMENT"_, "C_MKTSEGMENT"_)),
                                                "Where"_("StringContainsQ"_("C_MKTSEGMENT"_,
                                                                            "BUILDING"))),
                                      "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
                                  "Select"_(
                                      "Project"_(shallowCopy(orders),
                                                 "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                       "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                       "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                                      "Where"_(
                                          "Greater"_("DateObject"_("1995-03-15"), "O_ORDERDATE"_))),
                                  "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                              "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                    "O_CUSTKEY"_, "O_CUSTKEY"_, "O_SHIPPRIORITY"_,
                                    "O_SHIPPRIORITY"_)),
                          "Project"_(
                              "Select"_(
                                  "Project"_(shallowCopy(lineitem),
                                             "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_,
                                                   "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                   "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                  "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1993-03-15")))),
                              "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                    "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                          "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                      "As"_("Expr1009"_,
                            "Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)), // NOLINT
                            "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                            "O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                  "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_),
                  "As"_("revenue"_, "Sum"_("Expr1009"_))),
              "By"_("revenue"_, "desc"_, "O_ORDERDATE"_), 10); // NOLINT
        },
        []() { return "Dummy"_(); }},
       {"Q3 Post-Filter",
        [&]() {
          return "Top"_(
              "Group"_(
                  "Project"_(
                      "Select"_(
                          "Select"_(
                              "Select"_(
                                  "Project"_(
                                      "Join"_(
                                          "Project"_(
                                              "Join"_(
                                                  "Project"_(shallowCopy(customer),
                                                             "As"_("C_CUSTKEY"_, "C_CUSTKEY"_,
                                                                   "C_MKTSEGMENT"_,
                                                                   "C_MKTSEGMENT"_)),
                                                  "Project"_(shallowCopy(orders),
                                                             "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                                   "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                                   "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                                   "O_SHIPPRIORITY"_,
                                                                   "O_SHIPPRIORITY"_)),
                                                  "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                                              "As"_("C_MKTSEGMENT"_, "C_MKTSEGMENT"_, "O_ORDERKEY"_,
                                                    "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                    "O_CUSTKEY"_, "O_CUSTKEY"_, "O_SHIPPRIORITY"_,
                                                    "O_SHIPPRIORITY"_)),
                                          "Project"_(shallowCopy(lineitem),
                                                     "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                           "L_DISCOUNT"_, "L_DISCOUNT"_,
                                                           "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                           "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                          "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                                      "As"_("L_SHIPDATE"_, "L_SHIPDATE"_, "L_ORDERKEY"_,
                                            "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                            "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "O_ORDERKEY"_,
                                            "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                            "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_, "C_MKTSEGMENT"_,
                                            "C_MKTSEGMENT"_)),
                                  "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1993-03-15")))),
                              "Where"_("Greater"_("DateObject"_("1995-03-15"), "O_ORDERDATE"_))),
                          "Where"_("StringContainsQ"_("C_MKTSEGMENT"_, "BUILDING"))),
                      "As"_("expr1009"_, "Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)),
                            "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                            "O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                  "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_),
                  "As"_("revenue"_, "Sum"_("expr1009"_))),
              "By"_("revenue"_, "desc"_, "O_ORDERDATE"_), 10);
        },
        []() { return "Dummy"_(); }},
       {"Q9 (No Strings)",
        [&]() {
          return "Order"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Project"_(
                              "Join"_(
                                  "Project"_(
                                      "Join"_(
                                          "Project"_(
                                              "Select"_("Project"_(shallowCopy(part),
                                                                   "As"_("P_PARTKEY"_, "P_PARTKEY"_,
                                                                         "P_RETAILPRICE"_,
                                                                         "P_RETAILPRICE"_)),
                                                        "Where"_("Equal"_("P_RETAILPRICE"_,
                                                                          100.01))), // NOLINT
                                              "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                          "Project"_(
                                              "Join"_(
                                                  "Project"_(
                                                      "Join"_("Project"_(shallowCopy(nation),
                                                                         "As"_("N_REGIONKEY"_,
                                                                               "N_REGIONKEY"_,
                                                                               "N_NATIONKEY"_,
                                                                               "N_NATIONKEY"_)),
                                                              "Project"_(shallowCopy(supplier),
                                                                         "As"_("S_SUPPKEY"_,
                                                                               "S_SUPPKEY"_,
                                                                               "S_NATIONKEY"_,
                                                                               "S_NATIONKEY"_)),
                                                              "Where"_("Equal"_("N_NATIONKEY"_,
                                                                                "S_NATIONKEY"_))),
                                                      "As"_("N_REGIONKEY"_, "N_REGIONKEY"_,
                                                            "S_SUPPKEY"_, "S_SUPPKEY"_)),
                                                  "Project"_(shallowCopy(partsupp),
                                                             "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                   "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                   "PS_SUPPLYCOST"_,
                                                                   "PS_SUPPLYCOST"_)),
                                                  "Where"_("Equal"_("S_SUPPKEY"_, "PS_SUPPKEY"_))),
                                              "As"_("N_REGIONKEY"_, "N_REGIONKEY"_, "PS_PARTKEY"_,
                                                    "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                    "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),
                                          "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                      "As"_("N_REGIONKEY"_, "N_REGIONKEY"_, "PS_PARTKEY"_,
                                            "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                            "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),
                                  "Project"_(shallowCopy(lineitem),
                                             "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                                   "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                   "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                   "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                   "L_QUANTITY"_)),
                                  "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                    "List"_("L_PARTKEY"_, "L_SUPPKEY"_)))),
                              "As"_("N_REGIONKEY"_, "N_REGIONKEY"_, "PS_SUPPLYCOST"_,
                                    "PS_SUPPLYCOST"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                    "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                    "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_)),
                          "Project"_(shallowCopy(orders), "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                                "O_ORDERDATE"_, "O_ORDERDATE"_)),
                          "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                      "As"_("nation"_, "N_REGIONKEY"_, "O_YEAR"_, "Year"_("O_ORDERDATE"_),
                            "amount"_,
                            "Minus"_("Times"_("L_EXTENDEDPRICE"_,
                                              "Minus"_(1.0, "L_DISCOUNT"_)), // NOLINT
                                     "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                  "By"_("nation"_, "O_YEAR"_), "Sum"_("amount"_)),
              "By"_("nation"_, "O_YEAR"_, "desc"_));
        },
        []() { return "Dummy"_(); }},
       {"Q9",
        [&]() {
          return "Order"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Project"_(
                              "Join"_(
                                  "Project"_(
                                      "Join"_(
                                          "Project"_(
                                              "Select"_(
                                                  "Project"_(shallowCopy(part),
                                                             "As"_("P_PARTKEY"_, "P_PARTKEY"_,
                                                                   "P_NAME"_, "P_NAME"_)),
                                                  "Where"_("StringContainsQ"_("P_NAME"_, "green"))),
                                              "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                          "Project"_(
                                              "Join"_(
                                                  "Project"_(
                                                      "Join"_("Project"_(shallowCopy(nation),
                                                                         "As"_("N_NAME"_, "N_NAME"_,
                                                                               "N_NATIONKEY"_,
                                                                               "N_NATIONKEY"_)),
                                                              "Project"_(shallowCopy(supplier),
                                                                         "As"_("S_SUPPKEY"_,
                                                                               "S_SUPPKEY"_,
                                                                               "S_NATIONKEY"_,
                                                                               "S_NATIONKEY"_)),
                                                              "Where"_("Equal"_("N_NATIONKEY"_,
                                                                                "S_NATIONKEY"_))),
                                                      "As"_("N_NAME"_, "N_NAME"_, "S_SUPPKEY"_,
                                                            "S_SUPPKEY"_)),
                                                  "Project"_(shallowCopy(partsupp),
                                                             "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                   "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                   "PS_SUPPLYCOST"_,
                                                                   "PS_SUPPLYCOST"_)),
                                                  "Where"_("Equal"_("S_SUPPKEY"_, "PS_SUPPKEY"_))),
                                              "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_,
                                                    "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                    "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),
                                          "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                      "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_, "PS_PARTKEY"_,
                                            "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                            "PS_SUPPLYCOST"_)),
                                  "Project"_(shallowCopy(lineitem),
                                             "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                                   "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                   "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                   "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                   "L_QUANTITY"_)),
                                  "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                    "List"_("L_PARTKEY"_, "L_SUPPKEY"_)))),
                              "As"_("N_NAME"_, "N_NAME"_, "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_,
                                    "L_ORDERKEY"_, "L_ORDERKEY"_, "L_EXTENDEDPRICE"_,
                                    "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                    "L_QUANTITY"_)),
                          "Project"_(shallowCopy(orders), "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                                "O_ORDERDATE"_, "O_ORDERDATE"_)),
                          "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                      "As"_("nation"_, "N_NAME"_, "O_YEAR"_, "Year"_("O_ORDERDATE"_), "amount"_,
                            "Minus"_("Times"_("L_EXTENDEDPRICE"_,
                                              "Minus"_(1.0, "L_DISCOUNT"_)), // NOLINT
                                     "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                  "By"_("nation"_, "O_YEAR"_), "Sum"_("amount"_)),
              "By"_("nation"_, "O_YEAR"_, "desc"_));
        },
        []() { return "Dummy"_(); }},
       {"Q9 (AF Heuristics)",
        [&]() {
          return "Order"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Project"_(
                              "Select"_(
                                  "Project"_(
                                      "Join"_(
                                          "Project"_(shallowCopy(part),
                                                     "As"_("P_PARTKEY"_, "P_PARTKEY"_,
                                                           "P_RETAILPRICE"_, "P_RETAILPRICE"_)),
                                          "Project"_(
                                              "Join"_(
                                                  "Project"_(
                                                      "Join"_("Project"_(shallowCopy(nation),
                                                                         "As"_("N_NAME"_, "N_NAME"_,
                                                                               "N_NATIONKEY"_,
                                                                               "N_NATIONKEY"_)),
                                                              "Project"_(shallowCopy(supplier),
                                                                         "As"_("S_SUPPKEY"_,
                                                                               "S_SUPPKEY"_,
                                                                               "S_NATIONKEY"_,
                                                                               "S_NATIONKEY"_)),
                                                              "Where"_("Equal"_("N_NATIONKEY"_,
                                                                                "S_NATIONKEY"_))),
                                                      "As"_("N_NAME"_, "N_NAME"_, "S_SUPPKEY"_,
                                                            "S_SUPPKEY"_)),
                                                  "Project"_(shallowCopy(partsupp),
                                                             "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                   "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                   "PS_SUPPLYCOST"_,
                                                                   "PS_SUPPLYCOST"_)),
                                                  "Where"_("Equal"_("S_SUPPKEY"_, "PS_SUPPKEY"_))),
                                              "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_,
                                                    "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                    "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),
                                          "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                      "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_, "PS_PARTKEY"_,
                                            "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                            "PS_SUPPLYCOST"_, "P_RETAILPRICE"_, "P_RETAILPRICE"_)),
                                  "Where"_("Equal"_("P_RETAILPRICE"_, 100.01))), // NOLINT
                              "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_, "PS_PARTKEY"_,
                                    "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                    "PS_SUPPLYCOST"_)),
                          "Project"_(
                              "Join"_("Project"_(shallowCopy(orders),
                                                 "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                       "O_ORDERDATE"_)),
                                      "Project"_(shallowCopy(lineitem),
                                                 "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                                       "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                       "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                       "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                       "L_QUANTITY"_)),
                                      "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                              "As"_("O_ORDERDATE"_, "O_ORDERDATE"_, "L_EXTENDEDPRICE"_,
                                    "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                    "L_QUANTITY"_, "L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                    "L_SUPPKEY"_)),
                          "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                            "List"_("L_PARTKEY"_, "L_SUPPKEY"_)))),
                      "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                            "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)),
                                     "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                  "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
              "By"_("nation"_, "o_year"_, "desc"_));
        },
        []() { return "Dummy"_(); }},
       {"Q18 (No Strings)",
        [&]() {
          return "Top"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Select"_("Group"_("Project"_(shallowCopy(lineitem),
                                                        "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                              "L_QUANTITY"_, "L_QUANTITY"_)),
                                             "By"_("L_ORDERKEY"_),
                                             "As"_("sum_l_quantity"_, "Sum"_("L_QUANTITY"_))),
                                    "Where"_("Greater"_("sum_l_quantity"_, 1.0))), // NOLINT
                          "Project"_(
                              "Join"_("Project"_(shallowCopy(customer),
                                                 "As"_("C_ACCTBAL"_, "C_ACCTBAL"_, "C_CUSTKEY"_,
                                                       "C_CUSTKEY"_)),
                                      "Project"_(shallowCopy(orders),
                                                 "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_,
                                                       "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                       "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                                      "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                              "As"_("C_ACCTBAL"_, "C_ACCTBAL"_, "O_ORDERKEY"_, "O_ORDERKEY"_,
                                    "O_CUSTKEY"_, "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                    "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                          "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                      "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                            "O_TOTALPRICE"_, "O_TOTALPRICE"_, "C_ACCTBAL"_, "C_ACCTBAL"_,
                            "O_CUSTKEY"_, "O_CUSTKEY"_, "sum_l_quantity"_, "sum_l_quantity"_)),
                  "By"_("C_ACCTBAL"_, "O_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
                  "Sum"_("sum_l_quantity"_)),
              "By"_("O_TOTALPRICE"_, "desc"_, "O_ORDERDATE"_), 100); // NOLINT
        },
        []() { return "Dummy"_(); }},
       {"Q18",
        [&]() {
          return "Top"_(
              "Group"_(
                  "Project"_(
                      "Join"_(
                          "Select"_("Group"_("Project"_(shallowCopy(lineitem),
                                                        "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                              "L_QUANTITY"_, "L_QUANTITY"_)),
                                             "By"_("L_ORDERKEY"_),
                                             "As"_("sum_l_quantity"_, "Sum"_("L_QUANTITY"_))),
                                    "Where"_("Greater"_("sum_l_quantity"_, 1.0))), // NOLINT
                          "Project"_(
                              "Join"_("Project"_(
                                          shallowCopy(customer),
                                          "As"_("C_NAME"_, "C_NAME"_, "C_CUSTKEY"_, "C_CUSTKEY"_)),
                                      "Project"_(shallowCopy(orders),
                                                 "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_,
                                                       "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                       "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                                      "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                              "As"_("C_NAME"_, "C_NAME"_, "O_ORDERKEY"_, "O_ORDERKEY"_,
                                    "O_CUSTKEY"_, "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                    "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                          "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                      "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                            "O_TOTALPRICE"_, "O_TOTALPRICE"_, "C_NAME"_, "C_NAME"_, "O_CUSTKEY"_,
                            "O_CUSTKEY"_, "sum_l_quantity"_, "sum_l_quantity"_)),
                  "By"_("C_NAME"_, "O_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
                  "Sum"_("sum_l_quantity"_)),
              "By"_("O_TOTALPRICE"_, "desc"_, "O_ORDERDATE"_), 100); // NOLINT
        },
        []() { return "Dummy"_(); }}}));

  DYNAMIC_SECTION(queryName << (useCache ? " - with cache" : " - no cache")
                            << (multipleSpans ? " - multiple spans" : " - single span")) {
    auto output1 = eval(query());
    CHECK(output1 == expectedOutput());

    auto output2 = eval(query());
    CHECK(output2 == expectedOutput());

    auto output3 = eval(query());
    CHECK(output3 == expectedOutput());
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Summation of numeric Spans", "[spans]", std::int32_t, std::int64_t,
                   std::double_t) {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto input = GENERATE(take(3, chunk(50, random<TestType>(1, 1000))));
  auto sum = std::accumulate(begin(input), end(input), TestType());

  if constexpr(std::is_same_v<TestType, std::double_t>) {
    auto result = eval("Plus"_(boss::Span<TestType>(vector(input))));
    CHECK(get<std::double_t>(result) == Catch::Detail::Approx((std::double_t)sum));
  } else {
    auto result = eval("Plus"_(boss::Span<TestType>(vector(input))));
    CHECK(get<TestType>(result) == sum);
  }
}

auto createInt64SpanOf = [](auto... values) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<int64_t> v = {values...};
  auto s = boss::Span<int64_t>(std::move(v));
  SpanArguments args;
  args.emplace_back(std::move(s));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

using intType = int32_t;

auto createIntSpanOf = [](auto... values) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<intType> v = {values...};
  auto s = boss::Span<intType>(std::move(v));
  SpanArguments args;
  args.emplace_back(std::move(s));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

auto createFloatSpanOf = [](auto... values) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<double_t> v = {values...};
  auto s = boss::Span<double_t>(std::move(v));
  SpanArguments args;
  args.emplace_back(std::move(s));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

auto createStringSpanOf = [](auto... values) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<std::string> v = {values...};
  auto s = boss::Span<std::string>(std::move(v));
  SpanArguments args;
  args.emplace_back(std::move(s));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

TEST_CASE("Plus, Divide and Times atoms", "[hazard-adaptive-engine]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Plus") {
    CHECK(get<std::int32_t>(eval("Plus"_(5, 4))) == 9); // NOLINT
    CHECK(get<std::int32_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_("Plus"_(2, 3), 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval("Plus"_("Plus"_(3, 2), 2, 2))) == 9);
    CHECK(get<std::int32_t>(eval(9)) == 9);
  }

  SECTION("Divide") {
    CHECK(get<std::int32_t>(eval("Divide"_(12, 4))) == 3); // NOLINT
    CHECK(get<std::int32_t>(eval("Divide"_("Divide"_(12, 3), 2))) == 2);
  }

  SECTION("Times") {
    CHECK(get<std::int32_t>(eval("Times"_(5, 4))) == 20); // NOLINT
    CHECK(get<std::int32_t>(eval("Times"_(5, 2, 2))) == 20);
    CHECK(get<std::int32_t>(eval("Times"_(5, 2, 2))) == 20);
    CHECK(get<std::int32_t>(eval("Times"_("Times"_(2, 3), 2, 2))) == 24);
    CHECK(get<std::int32_t>(eval("Times"_("Times"_(3, 2), 2, 2))) == 24);
  }

  SECTION("Times and Plus") {
    CHECK(get<std::int32_t>(eval("Plus"_("Times"_(2, 3), 2, 2))) == 10);
    CHECK(get<std::int32_t>(eval("Times"_("Plus"_(3, 2), 2, 2))) == 20);
  }

  SECTION("Plus (double)") {
    auto const twoAndAHalf = 2.5F;
    auto const two = 2.0F;
    auto const quantum = 0.001F;
    CHECK(std::fabs(get<float>(eval("Plus"_(twoAndAHalf, twoAndAHalf))) - two * twoAndAHalf) <
          quantum);
  }

  SECTION("Times (double)") {
    auto const four = 4.0F;
    auto const two = 2.0F;
    auto const quantum = 0.001F;
    CHECK(std::fabs(get<float>(eval("Times"_(two, two))) - four) < quantum);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Plus spans", "[hazard-adaptive-engine]", std::int32_t, std::int64_t,
                   std::double_t) {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto input = GENERATE(take(3, chunk(50, random<TestType>(1, 1000))));
  auto sum = std::accumulate(begin(input), end(input), TestType());

  if constexpr(std::is_same_v<TestType, std::double_t>) {
    auto result = eval("Plus"_(boss::Span<TestType>(vector(input))));
    CHECK(get<std::double_t>(result) == Catch::Detail::Approx((std::double_t)sum));
  } else {
    auto result = eval("Plus"_(boss::Span<TestType>(vector(input))));
    CHECK(get<TestType>(result) == sum);
  }
}

TEST_CASE("Select", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Selection single column 1") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Equal"_("Value"_, 1))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5, 3, 1, 4, 1))), 2, 4));
#else
    CHECK(result == "Table"_("Value"_("List"_(1, 1))));
#endif
  }

  SECTION("Selection single column 2") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_("Value"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5, 3, 1, 4, 1))), 0, 3));
#else
    CHECK(result == "Table"_("Value"_("List"_(5, 4))));
#endif
  }

  SECTION("Selection single column 3") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_(3, "Value"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5, 3, 1, 4, 1))), 2, 4));
#else
    CHECK(result == "Table"_("Value"_("List"_(1, 1))));
#endif
  }

  SECTION("Empty selection 1") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Equal"_("Value"_, 6))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5, 3, 1, 4, 1)))));
#else
    CHECK(result == "Table"_("Value"_("List"_())));
#endif
  }

  SECTION("Selection multiple columns 1") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_("Value1"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)), "Value2"_("List"_(1, 2, 3, 4, 5))),
                    0, 3));
#else
    CHECK(result == "Table"_("Value1"_("List"_(5, 4)), "Value2"_("List"_(1, 4))));
#endif
  }

  SECTION("Selection multiple columns 2") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_("Value2"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)), "Value2"_("List"_(1, 2, 3, 4, 5))),
                    3, 4));
#else
    CHECK(result == "Table"_("Value1"_("List"_(4, 1)), "Value2"_("List"_(4, 5))));
#endif
  }

  SECTION("Selection nested 1") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_("Select"_(std::move(intTable), "Where"_("Greater"_("Value1"_, 1))),
                                 "Where"_("Greater"_("Value2"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Select"_("Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)),
                                                 "Value2"_("List"_(1, 2, 3, 4, 5))),
                                        0, 1, 3),
                              "Where"_("Greater"_("Value2"_, 3))));
#else
    CHECK(result == "Table"_("Value1"_("List"_(4)), "Value2"_("List"_(4))));
#endif
  }

  SECTION("Selection nested 2") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(2, 5, 4, 3, 1)), "Value2"_(createIntSpanOf(1, 3, 3, 4, 3))); // NOLINT
    auto result = eval("Select"_("Select"_(std::move(intTable), "Where"_("Equal"_("Value2"_, 3))),
                                 "Where"_("Greater"_("Value1"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Select"_("Gather"_("Table"_("Value1"_("List"_(2, 5, 4, 3, 1)),
                                                 "Value2"_("List"_(1, 3, 3, 4, 3))),
                                        1, 2, 4),
                              "Where"_("Greater"_("Value1"_, 3))));
#else
    CHECK(result == "Table"_("Value1"_("List"_(5, 4)), "Value2"_("List"_(3, 3))));
#endif
  }

  SECTION("Empty selection 2") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(2, 5, 4, 3, 1)), "Value2"_(createIntSpanOf(1, 3, 3, 4, 3))); // NOLINT
    auto result = eval("Select"_("Select"_(std::move(intTable), "Where"_("Equal"_("Value2"_, 6))),
                                 "Where"_("Greater"_("Value1"_, 3))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Select"_("Gather"_("Table"_("Value1"_("List"_(2, 5, 4, 3, 1)),
                                                 "Value2"_("List"_(1, 3, 3, 4, 3)))),
                              "Where"_("Greater"_("Value1"_, 3))));
#else
    CHECK(result == "Table"_("Value1"_("List"_()), "Value2"_("List"_())));
#endif
  }
}

TEST_CASE("Select Spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto intTable = "Table"_("Value"_("List"_(boss::Span<int64_t>({5, 3, 1, 4, 1})))); // NOLINT

  SECTION("Selection") {
    auto result = eval("Select"_(std::move(intTable), "Where"_("Greater"_("Value"_, 3L))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5L, 3L, 1L, 4L, 1L))), 0, 3));
#else
    CHECK(result == "Table"_("Value"_("List"_(5L, 4L))));
#endif
  }
}

TEST_CASE("Project", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto customerTable = "Table"_("FirstName"_(createStringSpanOf("John", "Sam", "Barbara")),
                                "LastName"_(createStringSpanOf("McCarthy", "Madden", "Liskov")));

  SECTION("Projection") {
    auto fullnames = eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                                     "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
    CHECK(fullnames == "Table"_("FirstName"_("List"_("John", "Sam", "Barbara")),
                                "LastName"_("List"_("McCarthy", "Madden", "Liskov"))));
    auto firstNames = eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                                      "As"_("FirstName"_, "FirstName"_)));
    CHECK(firstNames == "Table"_("FirstName"_("List"_("John", "Sam", "Barbara"))));
    auto lastNames = eval(
        "Project"_(customerTable.clone(CloneReason::FOR_TESTING), "As"_("LastName"_, "LastName"_)));
    CHECK(lastNames == "Table"_("LastName"_("List"_("McCarthy", "Madden", "Liskov"))));
    auto updatedNames =
        eval("Project"_(customerTable.clone(CloneReason::FOR_TESTING),
                        "As"_("NewFirstName"_, "FirstName"_, "NewLastName"_, "LastName"_)));
    CHECK(updatedNames == "Table"_("NewFirstName"_("List"_("John", "Sam", "Barbara")),
                                   "NewLastName"_("List"_("McCarthy", "Madden", "Liskov"))));
  }
}

TEST_CASE("Plus, Times and Divide Cols", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto intTable = "Table"_("Value"_(createIntSpanOf(10, 20, 30, 40, 50))); // NOLINT

  SECTION("Plus") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Plus"_("Value"_, 5)))) ==
          "Table"_("Result"_(createIntSpanOf(15, 25, 35, 45, 55)))); // NOLINT
  }

  SECTION("Times") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Times"_("Value"_, 2)))) ==
          "Table"_("Result"_("List"_(20, 40, 60, 80, 100)))); // NOLINT8
  }

  SECTION("Divide") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Divide"_("Value"_, 10)))) ==
          "Table"_("Result"_("List"_(1, 2, 3, 4, 5)))); // NOLINT8
  }

  SECTION("Minus") {
    CHECK(get<std::int32_t>(eval("Minus"_(5, 4))) == 1);

    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Minus"_("Value"_, 10)))) ==
          "Table"_("Result"_("List"_(0, 10, 20, 30, 40)))); // NOLINT8

    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Minus"_(0, "Value"_)))) ==
          "Table"_("Result"_("List"_(-10, -20, -30, -40, -50)))); // NOLINT8
  }

  SECTION("Plus and Times") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Plus"_("Times"_("Value"_, 2), 5)))) ==
          "Table"_("Result"_("List"_(25, 45, 65, 85, 105)))); // NOLINT

    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Times"_("Plus"_("Value"_, 5), 2)))) ==
          "Table"_("Result"_("List"_(30, 50, 70, 90, 110)))); // NOLINT8
  }

  SECTION("Times and minus") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Minus"_("Times"_("Value"_, 2), 5)))) ==
          "Table"_("Result"_("List"_(15, 35, 55, 75, 95)))); // NOLINT
  }
}

TEST_CASE("Select multiple predicates", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Selection multiple columns 1") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(
        std::move(intTable), "Where"_("And"_("Greater"_("Value1"_, 3), "Greater"_(5, "Value1"_)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)), "Value2"_("List"_(1, 2, 3, 4, 5))),
                    3));
#else
    CHECK(result == "Table"_("Value1"_("List"_(4)), "Value2"_("List"_(4))));
#endif
  }

  SECTION("Selection multiple columns 2") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable),
                                 "Where"_("And"_("Greater"_("Value2"_, 1), "Greater"_(5, "Value2"_),
                                                 "Greater"_("Value1"_, 2)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)), "Value2"_("List"_(1, 2, 3, 4, 5))),
                    1, 3));
#else
    CHECK(result == "Table"_("Value1"_("List"_(3, 4)), "Value2"_("List"_(2, 4))));
#endif
  }

  SECTION("Empty selection 3") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable),
                                 "Where"_("And"_("Greater"_("Value2"_, 6), "Greater"_(5, "Value2"_),
                                                 "Greater"_("Value1"_, 2)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)),
                                       "Value2"_("List"_(1, 2, 3, 4, 5)))));
#else
    CHECK(result == "Table"_("Value1"_("List"_()), "Value2"_("List"_())));
#endif
  }

  SECTION("Empty selection 4") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable),
                                 "Where"_("And"_("Greater"_("Value2"_, 1), "Greater"_(0, "Value2"_),
                                                 "Greater"_("Value1"_, 2)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value1"_("List"_(5, 3, 1, 4, 1)),
                                       "Value2"_("List"_(1, 2, 3, 4, 5)))));
#else
    CHECK(result == "Table"_("Value1"_("List"_()), "Value2"_("List"_())));
#endif
  }
}

TEST_CASE("Plus, Divide and Times 2", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto intTable = "Table"_("Value"_(createIntSpanOf(10, 20, 30, 40, 50))); // NOLINT

  SECTION("Plus") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Plus"_("Value"_, "Value"_)))) ==
          "Table"_("Result"_("List"_(20, 40, 60, 80, 100)))); // NOLINT
  }

  SECTION("Plus") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Plus"_("Value"_, "Value"_, "Value"_)))) ==
          "Table"_("Result"_("List"_(30, 60, 90, 120, 150)))); // NOLINT
  }

  SECTION("Times") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Times"_("Value"_, "Value"_)))) ==
          "Table"_("Result"_("List"_(100, 400, 900, 1600, 2500)))); // NOLINT
  }

  SECTION("Divide") {
    CHECK(eval("Project"_(intTable.clone(CloneReason::FOR_TESTING),
                          "As"_("Result"_, "Divide"_("Value"_, "Value"_)))) ==
          "Table"_("Result"_("List"_(1, 1, 1, 1, 1)))); // NOLINT
  }
}

TEST_CASE("Project with calculation", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto lineitem =
      "Table"_("L_ORDERKEY"_(createIntSpanOf(1, 1, 2, 3)),                                 // NOLINT
               "L_PARTKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_SUPPKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_RETURNFLAG"_(createStringSpanOf("N", "N", "A", "A")),                       // NOLINT
               "L_LINESTATUS"_(createStringSpanOf("O", "O", "F", "F")),                       // NOLINT
               "L_RETURNFLAG_INT"_(createInt64SpanOf('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
               "L_LINESTATUS_INT"_(createInt64SpanOf('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
               "L_QUANTITY"_(createIntSpanOf(17, 21, 8, 5)),                               // NOLINT
               "L_EXTENDEDPRICE"_(createFloatSpanOf(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
               "L_DISCOUNT"_(createFloatSpanOf(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
               "L_TAX"_(createFloatSpanOf(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
               "L_SHIPDATE"_(createIntSpanOf(8400, 9130, 9861, 9130)));

  SECTION("Project with calc") {
    auto output = eval("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                  "As"_("calc1"_, "Plus"_("L_ORDERKEY"_, 1))));
    CHECK(output == "Table"_("calc1"_("List"_(2, 2, 3, 4)))); // NOLINT
  }
}

TEST_CASE("Dates and Group", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto lineitem =
      "Table"_("L_ORDERKEY"_(createIntSpanOf(1, 1, 2, 3)),                                 // NOLINT
               "L_PARTKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_SUPPKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_RETURNFLAG"_(createStringSpanOf("N", "N", "A", "A")),                       // NOLINT
               "L_LINESTATUS"_(createStringSpanOf("O", "O", "F", "F")),                       // NOLINT
               "L_RETURNFLAG_INT"_(createInt64SpanOf('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
               "L_LINESTATUS_INT"_(createInt64SpanOf('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
               "L_QUANTITY"_(createIntSpanOf(17, 21, 8, 5)),                               // NOLINT
               "L_EXTENDEDPRICE"_(createFloatSpanOf(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
               "L_DISCOUNT"_(createFloatSpanOf(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
               "L_TAX"_(createFloatSpanOf(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
               "L_SHIPDATE"_(createIntSpanOf(8400, 9130, 9861, 9130)));

  SECTION("Dates") {
    auto output = eval("Project"_(
        "Select"_("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                             "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_SHIPDATE"_, "L_SHIPDATE"_)),
                  "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1995-01-01")))),
        "As"_("L_QUANTITY"_, "L_QUANTITY"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output == "Project"_("Gather"_("Table"_("L_QUANTITY"_("List"_(17, 21, 8, 5)),
                                                  "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
                                         2),
                               "As"_("L_QUANTITY"_, "L_QUANTITY"_)));
#else
    CHECK(output == "Table"_("L_QUANTITY"_("List"_(8)))); // NOLINT
#endif
  }

  SECTION("Group 1") {
    auto output = eval("Group"_(lineitem.clone(CloneReason::FOR_TESTING), "Sum"_("L_DISCOUNT"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "Sum"_("L_DISCOUNT"_)));
#else
    CHECK(output == "Table"_("L_DISCOUNT"_("List"_(0.27)))); // NOLINT
#endif
  }

  SECTION("Group 2") {
    auto output = eval("Group"_(lineitem.clone(CloneReason::FOR_TESTING), "Sum"_("L_DISCOUNT"_),
                                "Count"_("L_DISCOUNT"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "Sum"_("L_DISCOUNT"_), "Count"_("L_DISCOUNT"_)));
#else
    CHECK(output == "Table"_("L_DISCOUNT"_("List"_(0.27)),
                             "L_DISCOUNT"_("List"_(4)))); // NOLINT
#endif
  }

  SECTION("Group 3") {
    auto output = eval("Group"_(lineitem.clone(CloneReason::FOR_TESTING),
                                "As"_("total_discount"_, "Sum"_("L_DISCOUNT"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "As"_("total_discount"_, "Sum"_("L_DISCOUNT"_))));
#else
    CHECK(output == "Table"_("total_discount"_("List"_(0.27)))); // NOLINT
#endif
  }

  SECTION("Group 4") {
    auto output = eval("Group"_(lineitem.clone(CloneReason::FOR_TESTING), "By"_("L_ORDERKEY"_),
                                "As"_("sumParts"_, "Sum"_("L_PARTKEY"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "By"_("L_ORDERKEY"_), "As"_("sumParts"_, "Sum"_("L_PARTKEY"_))));
#else
    CHECK(output ==
          "Table"_("L_ORDERKEY"_("List"_(1, 2, 3)), "sumParts"_("List"_(3, 3, 4)))); // NOLINT
#endif
  }

  SECTION("Group 5") {
    auto output = eval("Group"_(lineitem.clone(CloneReason::FOR_TESTING), "By"_("L_ORDERKEY"_),
                                "As"_("count"_, "Count"_("L_ORDERKEY"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "By"_("L_ORDERKEY"_), "As"_("count"_, "Count"_("L_ORDERKEY"_))));
#else
    CHECK(output ==
          "Table"_("L_ORDERKEY"_("List"_(1, 2, 3)), "count"_("List"_(2, 1, 1)))); // NOLINT
#endif
  }

  SECTION("Group 6") {
    auto output = eval(
        "Group"_(lineitem.clone(CloneReason::FOR_TESTING), "By"_("L_ORDERKEY"_),
                 "As"_("sum_quantity"_, "Sum"_("L_QUANTITY"_), "count"_, "Count"_("L_ORDERKEY"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "By"_("L_ORDERKEY"_),
              "As"_("sum_quantity"_, "Sum"_("L_QUANTITY"_), "count"_, "Count"_("L_ORDERKEY"_))));
#else
    CHECK(output == "Table"_("L_ORDERKEY"_("List"_(1, 2, 3)), "sum_quantity"_("List"_(38, 8, 5)),
                             "count"_("List"_(2, 1, 1)))); // NOLINT
#endif
  }

  SECTION("Group 7") {
    auto output = eval("Group"_(
        lineitem.clone(CloneReason::FOR_TESTING), "By"_("L_ORDERKEY"_),
        "As"_("sum_price"_, "Sum"_("L_EXTENDEDPRICE"_), "count"_, "Count"_("L_ORDERKEY"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_(
              "Table"_("L_ORDERKEY"_("List"_(1, 1, 2, 3)),                                 // NOLINT
                       "L_PARTKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_SUPPKEY"_("List"_(1, 2, 3, 4)),                                  // NOLINT
                       "L_RETURNFLAG"_("List"_("N", "N", "A", "A")),                       // NOLINT
                       "L_LINESTATUS"_("List"_("O", "O", "F", "F")),                       // NOLINT
                       "L_RETURNFLAG_INT"_("List"_('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
                       "L_LINESTATUS_INT"_("List"_('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
                       "L_QUANTITY"_("List"_(17, 21, 8, 5)),                               // NOLINT
                       "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
                       "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
                       "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
                       "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130))),
              "By"_("L_ORDERKEY"_),
              "As"_("sum_price"_, "Sum"_("L_EXTENDEDPRICE"_), "count"_, "Count"_("L_ORDERKEY"_))));
#else
    CHECK(output == "Table"_("L_ORDERKEY"_("List"_(1, 2, 3)),
                             "sum_price"_("List"_(17954.55 + 34850.16, 7712.48, 25284.00)),
                             "count"_("List"_(2, 1, 1)))); // NOLINT
#endif
  }
}

TEST_CASE("TPC-H Q6", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto lineitem =
      "Table"_("L_ORDERKEY"_(createIntSpanOf(1, 1, 2, 3)),                                 // NOLINT
               "L_PARTKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_SUPPKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_RETURNFLAG"_(createStringSpanOf("N", "N", "A", "A")),                       // NOLINT
               "L_LINESTATUS"_(createStringSpanOf("O", "O", "F", "F")),                       // NOLINT
               "L_RETURNFLAG_INT"_(createInt64SpanOf('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
               "L_LINESTATUS_INT"_(createInt64SpanOf('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
               "L_QUANTITY"_(createIntSpanOf(17, 21, 8, 5)),                               // NOLINT
               "L_EXTENDEDPRICE"_(createFloatSpanOf(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
               "L_DISCOUNT"_(createFloatSpanOf(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
               "L_TAX"_(createFloatSpanOf(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
               "L_SHIPDATE"_(createIntSpanOf(8400, 9130, 9861, 9130)));

  SECTION("Q6 (No Grouping)") {
    auto output = eval("Project"_(
        "Select"_(
            "Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                       "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                             "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
            "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),      // NOLINT
                            "Greater"_("L_DISCOUNT"_, 0.0499),  // NOLINT
                            "Greater"_(0.07001, "L_DISCOUNT"_), // NOLINT
                            "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                            "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
        "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Project"_("Gather"_("Table"_("L_QUANTITY"_("List"_(17, 21, 8, 5)),           // NOLINT
                                        "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)), // NOLINT
                                        "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130)), // NOLINT
                                        "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48,
                                                                   25284.00))) // NOLINT
                               ,
                               1, 3),
                     "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))));
#else
    CHECK(output == "Table"_("revenue"_("List"_(34850.16 * 0.05, 25284.00 * 0.06)))); // NOLINT
#endif
  }

  SECTION("Q6") {
    auto output = eval("Group"_(
        "Project"_(
            "Select"_("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                       "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_,
                                       "L_EXTENDEDPRICE"_)),
                      "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),      // NOLINT
                                      "Greater"_("L_DISCOUNT"_, 0.0499),  // NOLINT
                                      "Greater"_(0.07001, "L_DISCOUNT"_), // NOLINT
                                      "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                      "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
            "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
        "Sum"_("revenue"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output ==
          "Group"_("Project"_(
                       "Gather"_("Table"_("L_QUANTITY"_("List"_(17, 21, 8, 5)),           // NOLINT
                                          "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)), // NOLINT
                                          "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130)), // NOLINT
                                          "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48,
                                                                     25284.00))) // NOLINT
                                 ,
                                 1, 3),
                       "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
                   "Sum"_("revenue"_)));
#else
    CHECK(output == "Table"_("revenue"_("List"_(34850.16 * 0.05 + 25284.00 * 0.06)))); // NOLINT
#endif
  }
}

TEST_CASE("TPC-H Q1", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto lineitem =
      "Table"_("L_ORDERKEY"_(createIntSpanOf(1, 1, 2, 3)),                                 // NOLINT
               "L_PARTKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_SUPPKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_RETURNFLAG"_(createStringSpanOf("N", "N", "A", "A")),                       // NOLINT
               "L_LINESTATUS"_(createStringSpanOf("O", "O", "F", "F")),                       // NOLINT
               "L_RETURNFLAG_INT"_(createInt64SpanOf('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
               "L_LINESTATUS_INT"_(createInt64SpanOf('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
               "L_QUANTITY"_(createIntSpanOf(17, 21, 8, 5)),                               // NOLINT
               "L_EXTENDEDPRICE"_(createFloatSpanOf(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
               "L_DISCOUNT"_(createFloatSpanOf(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
               "L_TAX"_(createFloatSpanOf(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
               "L_SHIPDATE"_(createIntSpanOf(8400, 9130, 9861, 9130)));

  SECTION("Q1 (Project only)") {
    auto output = eval("Project"_(
        "Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                         "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_("L_TAX"_, 1.0),
                         "L_DISCOUNT"_, "L_DISCOUNT"_)),
        "As"_("disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
              "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_QUANTITY"_, "L_QUANTITY"_,
              "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_)));
    CHECK(
        output ==
        "Table"_("disc_price"_("List"_(17954.55 * (1.0 - 0.10), 34850.16 * (1.0 - 0.05),  // NOLINT
                                       7712.48 * (1.0 - 0.06), 25284.00 * (1.0 - 0.06))), // NOLINT
                 "charge"_("List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0),                // NOLINT
                                   34850.16 * (1.0 - 0.05) * (0.06 + 1.0),                // NOLINT
                                   7712.48 * (1.0 - 0.06) * (0.02 + 1.0),                 // NOLINT
                                   25284.00 * (1.0 - 0.06) * (0.06 + 1.0))),              // NOLINT
                 "L_QUANTITY"_("List"_(17, 21, 8, 5)),                                    // NOLINT
                 "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)),      // NOLINT
                 "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06))));                        // NOLINT
  }

  SECTION("Q1 (Select-Project only)") {
    auto output = eval("Project"_(
        "Project"_(
            "Select"_("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                       "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_,
                                       "L_EXTENDEDPRICE"_, "L_TAX"_, "L_TAX"_)),
                      "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
            "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "calc1"_,
                  "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_("L_TAX"_, 1.0), "L_DISCOUNT"_,
                  "L_DISCOUNT"_)),
        "As"_("disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
              "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_QUANTITY"_, "L_QUANTITY"_,
              "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(
        output ==
        "Project"_(
            "Project"_("Gather"_("Table"_("L_QUANTITY"_("List"_(17, 21, 8, 5)),           // NOLINT
                                          "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06)), // NOLINT
                                          "L_SHIPDATE"_("List"_(8400, 9130, 9861, 9130)), // NOLINT
                                          "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48,
                                                                     25284.00)),     // NOLINT
                                          "L_TAX"_("List"_(0.02, 0.06, 0.02, 0.06))) // NOLINT
                                 ,
                                 0, 1, 2, 3),
                       "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                             "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                             "Plus"_("L_TAX"_, 1.0), "L_DISCOUNT"_, "L_DISCOUNT"_)),
            "As"_("disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                  "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_QUANTITY"_, "L_QUANTITY"_,
                  "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_)));
#else
    CHECK(
        output ==
        "Table"_("disc_price"_("List"_(17954.55 * (1.0 - 0.10), 34850.16 * (1.0 - 0.05),  // NOLINT
                                       7712.48 * (1.0 - 0.06), 25284.00 * (1.0 - 0.06))), // NOLINT
                 "charge"_("List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0),                // NOLINT
                                   34850.16 * (1.0 - 0.05) * (0.06 + 1.0),                // NOLINT
                                   7712.48 * (1.0 - 0.06) * (0.02 + 1.0),                 // NOLINT
                                   25284.00 * (1.0 - 0.06) * (0.06 + 1.0))),              // NOLINT
                 "L_QUANTITY"_("List"_(17, 21, 8, 5)),                                    // NOLINT
                 "L_EXTENDEDPRICE"_("List"_(17954.55, 34850.16, 7712.48, 25284.00)),      // NOLINT
                 "L_DISCOUNT"_("List"_(0.10, 0.05, 0.06, 0.06))));                        // NOLINT
#endif
  }
}

TEST_CASE("Project with calculation 2", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto lineitem =
      "Table"_("L_ORDERKEY"_(createIntSpanOf(1, 1, 2, 3)),                                 // NOLINT
               "L_PARTKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_SUPPKEY"_(createIntSpanOf(1, 2, 3, 4)),                                  // NOLINT
               "L_RETURNFLAG"_(createStringSpanOf("N", "N", "A", "A")),                       // NOLINT
               "L_LINESTATUS"_(createStringSpanOf("O", "O", "F", "F")),                       // NOLINT
               "L_RETURNFLAG_INT"_(createInt64SpanOf('N'_i64, 'N'_i64, 'A'_i64, 'A'_i64)),   // NOLINT
               "L_LINESTATUS_INT"_(createInt64SpanOf('O'_i64, 'O'_i64, 'F'_i64, 'F'_i64)),   // NOLINT
               "L_QUANTITY"_(createIntSpanOf(17, 21, 8, 5)),                               // NOLINT
               "L_EXTENDEDPRICE"_(createFloatSpanOf(17954.55, 34850.16, 7712.48, 25284.00)), // NOLINT
               "L_DISCOUNT"_(createFloatSpanOf(0.10, 0.05, 0.06, 0.06)),                     // NOLINT
               "L_TAX"_(createFloatSpanOf(0.02, 0.06, 0.02, 0.06)),                          // NOLINT
               "L_SHIPDATE"_(createIntSpanOf(8400, 9130, 9861, 9130)));

  SECTION("Project without calc") {
    auto output = eval("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                  "As"_("L_ORDERKEY_new"_, "L_ORDERKEY"_)));
    CHECK(output == "Table"_("L_ORDERKEY_new"_("List"_(1, 1, 2, 3)))); // NOLINT
  }

  SECTION("Project with calc") {
    auto output = eval("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                  "As"_("calc1"_, "Plus"_("L_ORDERKEY"_, 1))));
    CHECK(output == "Table"_("calc1"_("List"_(2, 2, 3, 4)))); // NOLINT
  }

  SECTION("Project with calc after project") {
    auto output = eval(
        "Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                   "As"_("L_ORDERKEY_new"_, "L_ORDERKEY"_, "calc1"_, "Plus"_("L_ORDERKEY"_, 1))));
    CHECK(output == "Table"_("L_ORDERKEY_new"_("List"_(1, 1, 2, 3)),
                             "calc1"_("List"_(2, 2, 3, 4)))); // NOLINT
  }
}

TEST_CASE("Not evaluated", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Select - no table") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval(
        "Select"_("Dummy"_, "Where"_("And"_("Greater"_("Value1"_, 3), "Greater"_(5, "Value1"_)))));
    CHECK(result == "Select"_("Dummy"_, "Where"_("And"_("Greater"_("Value1"_, 3),
                                                        "Greater"_(5, "Value1"_)))));
  }

  SECTION("Select - incorrectly named table") {
    auto result = eval(
        "Select"_("NotTable"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))),
                  "Where"_("And"_("Greater"_("Value1"_, 3), "Greater"_(5, "Value1"_)))));
    CHECK(
        result ==
        "Select"_("NotTable"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))),
                  "Where"_("And"_("Greater"_("Value1"_, 3), "Greater"_(5, "Value1"_)))));
  }

  SECTION("Select - no predicate") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Dummy"_)));
    CHECK(result ==
          "Select"_("Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))),
                    "Where"_("Dummy"_)));
  }

  SECTION("Projection") {
    auto result =
        eval("Project"_("Dummy"_, "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
    CHECK(result ==
          "Project"_("Dummy"_, "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
  }
}

TEST_CASE("Partially evaluated", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Evaluate project but not select") {
    auto intTable =
        "Table"_("Value1"_(createIntSpanOf(5, 3, 1, 4, 1)), "Value2"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto result = eval(
        "Select"_("Dummy"_, "Where"_("And"_("Greater"_("Value1"_, 3), "Greater"_(5, "Value1"_)))));
    CHECK(result == "Select"_("Dummy"_, "Where"_("And"_("Greater"_("Value1"_, 3),
                                                        "Greater"_(5, "Value1"_)))));
  }
}

TEST_CASE("Gather", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Simple gather 1") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_(std::move(intTable), "Where"_("Equal"_("Value"_, 1))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("Value"_("List"_(5, 3, 1, 4, 1))), 2, 4));
#else
    CHECK(result == "Table"_("Value"_("List"_(1, 1))));
#endif
  }

  SECTION("Simple gather 2") {
    auto intTable = "Table"_("Value"_(createIntSpanOf(5, 3, 1, 4, 1))); // NOLINT
    auto result = eval("Select"_("Project"_(std::move(intTable), "As"_("key"_, "Value"_)),
                                 "Where"_("Equal"_("key"_, 1))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result == "Gather"_("Table"_("key"_("List"_(5, 3, 1, 4, 1))), 2, 4));
#else
    CHECK(result == "Table"_("key"_("List"_(1, 1))));
#endif
  }
}

auto createTwoSpansIntStartingFrom = [](intType n) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<intType> v1 = {0 + n, 1 + n};
  std::vector<intType> v2 = {2 + n, 3 + n};
  auto s1 = boss::Span<intType>(std::move(v1));
  auto s2 = boss::Span<intType>(std::move(v2));
  SpanArguments args;
  args.emplace_back(std::move(s1));
  args.emplace_back(std::move(s2));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

auto createTwoSpansInt = [](intType n1, intType n2) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<intType> v1 = {n1, n1 + 1, n1 + 2};
  std::vector<intType> v2 = {n2, n2 + 1, n2 + 2};
  auto s1 = boss::Span<intType>(std::move(v1));
  auto s2 = boss::Span<intType>(std::move(v2));
  SpanArguments args;
  args.emplace_back(std::move(s1));
  args.emplace_back(std::move(s2));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

auto createTwoSpansIntDecrease = [](intType n1, intType n2) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<intType> v1 = {n1 + 2, n1 + 1, n1};
  std::vector<intType> v2 = {n2 + 2, n2 + 1, n2};
  auto s1 = boss::Span<intType>(std::move(v1));
  auto s2 = boss::Span<intType>(std::move(v2));
  SpanArguments args;
  args.emplace_back(std::move(s1));
  args.emplace_back(std::move(s2));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

auto createFourSpansIntFrom = [](intType n) {
  using SpanArguments = boss::expressions::ExpressionSpanArguments;
  std::vector<intType> v1 = {7 + n, 3 + n};
  std::vector<intType> v2 = {6 + n, 2 + n};
  std::vector<intType> v3 = {5 + n, 1 + n};
  std::vector<intType> v4 = {4 + n, 0 + n};
  auto s1 = boss::Span<intType>(std::move(v1));
  auto s2 = boss::Span<intType>(std::move(v2));
  auto s3 = boss::Span<intType>(std::move(v3));
  auto s4 = boss::Span<intType>(std::move(v4));
  SpanArguments args;
  args.emplace_back(std::move(s1));
  args.emplace_back(std::move(s2));
  args.emplace_back(std::move(s3));
  args.emplace_back(std::move(s4));
  return boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));
};

TEST_CASE("Project - multiple spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                        "payload"_(createTwoSpansIntStartingFrom(4)));

  SECTION("Projection - multiple spans") {
    auto updatedNames = eval(
        "Project"_(std::move(table), "As"_("NewKeyName"_, "key"_, "NewPayloadName"_, "payload"_)));
    CHECK(updatedNames ==
          "Table"_("NewKeyName"_("List"_(0, 1, 2, 3)), "NewPayloadName"_("List"_(4, 5, 6, 7))));
  }
}

TEST_CASE("Select - multiple spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                        "payload"_(createTwoSpansIntStartingFrom(4)));

  SECTION("Select - multiple spans") {
    auto result = eval("Select"_(std::move(table), "Where"_("Greater"_("key"_, 0))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(
        result ==
        "Gather"_("Table"_("key"_("List"_(0, 1, 2, 3)), "payload"_("List"_(4, 5, 6, 7))), 1, 0, 1));
#else
    CHECK(result == "Table"_("key"_("List"_(1, 2, 3)), "payload"_("List"_(5, 6, 7))));
#endif
  }
}

TEST_CASE("Select multiple predicates and spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Selection multiple columns and spans 1") {
    auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                          "payload"_(createTwoSpansIntStartingFrom(4)));
    auto result = eval("Select"_(std::move(table),
                                 "Where"_("And"_("Greater"_("key"_, 0), "Greater"_(3, "key"_)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("key"_("List"_(0, 1, 2, 3)), "payload"_("List"_(4, 5, 6, 7))), 1, 0));
#else
    CHECK(result == "Table"_("key"_("List"_(1, 2)), "payload"_("List"_(5, 6))));
#endif
  }

  SECTION("Selection multiple columns and spans 2") {
    auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                          "payload"_(createTwoSpansIntStartingFrom(4)));
    auto result = eval("Select"_(
        std::move(table), "Where"_("And"_("Greater"_("key"_, 0), "Greater"_(7, "payload"_)))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(result ==
          "Gather"_("Table"_("key"_("List"_(0, 1, 2, 3)), "payload"_("List"_(4, 5, 6, 7))), 1, 0));
#else
    CHECK(result == "Table"_("key"_("List"_(1, 2)), "payload"_("List"_(5, 6))));
#endif
  }
}

TEST_CASE("Plus - mulitple spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::forward<decltype(expression)>(expression)));
  };

  SECTION("Project with calc 1") {
    auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                          "payload"_(createTwoSpansIntStartingFrom(4)));
    auto output = eval("Project"_(std::move(table), "As"_("calc1"_, "Plus"_("key"_, 1))));
    CHECK(output == "Table"_("calc1"_("List"_(1, 2, 3, 4)))); // NOLINT
  }

  SECTION("Project with calc 2") {
    auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                          "payload"_(createTwoSpansIntStartingFrom(4)));
    auto output = eval("Project"_(std::move(table), "As"_("calc1"_, "Minus"_("key"_, 10))));
    CHECK(output == "Table"_("calc1"_("List"_(-10, -9, -8, -7)))); // NOLINT
  }
}

TEST_CASE("Group multiple spans", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  auto table = "Table"_("key"_(createTwoSpansIntStartingFrom(0)),
                        "payload"_(createTwoSpansIntStartingFrom(4)));

  SECTION("Group sum multiple spans") {
    auto output = eval("Group"_(table.clone(CloneReason::FOR_TESTING), "Sum"_("key"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output == "Group"_("Table"_("key"_("List"_(0, 1, 2, 3)), // NOLINT
                                      "payload"_("List"_(4, 5, 6, 7))), "Sum"_("key"_)));
#else
    CHECK(output == "Table"_("key"_("List"_(6)))); // NOLINT
#endif
  }

  SECTION("Group count multiple spans") {
    auto output = eval("Group"_(table.clone(CloneReason::FOR_TESTING), "Count"_("key"_)));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output == "Group"_("Table"_("key"_("List"_(0, 1, 2, 3)), // NOLINT
                                      "payload"_("List"_(4, 5, 6, 7))), "Count"_("key"_)));
#else
    CHECK(output == "Table"_("key"_("List"_(4)))); // NOLINT
#endif
  }

  SECTION("Group count multiple spans with grouping") {
    auto output = eval("Group"_(table.clone(CloneReason::FOR_TESTING), "By"_("key"_),
                                "As"_("countKeys"_, "Count"_("key"_))));
#ifdef DEFERRED_TO_OTHER_ENGINE
    CHECK(output == "Group"_("Table"_("key"_("List"_(0, 1, 2, 3)), // NOLINT
                                      "payload"_("List"_(4, 5, 6, 7))), "By"_("key"_),
                                      "As"_("countKeys"_, "Count"_("key"_))));
#else
    CHECK(output ==
          "Table"_("key"_("List"_(0, 1, 2, 3)), "countKeys"_("List"_(1, 1, 1, 1)))); // NOLINT
#endif
  }
}

TEST_CASE("Join - single-span", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Simple join 1") {
    auto intTable1 = "Table"_("L_key"_(createIntSpanOf(500, 2, 1, 984, 871)),
                              "L_value"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createIntSpanOf(7, 8, 1, 2)),
                              "O_value"_(createIntSpanOf(1, 2, 3, 4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5))),
                                            "Partition"_("L_key"_("List"_(1)),"Indexes"_(2)),
                                            "Partition"_("L_key"_("List"_(2)),"Indexes"_(1))),
                           "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4))),
                                            "Partition"_("O_key"_("List"_(1)),"Indexes"_(2)),
                                            "Partition"_("O_key"_("List"_(2)),"Indexes"_(3))),
                           "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 2") {
    auto intTable1 = "Table"_("L_key"_(createIntSpanOf(4000001, 4000002, 20009, 5, 4)),
                              "L_value"_(createIntSpanOf(1, 2, 3, 4, 5))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createIntSpanOf(5, 4, 20009, 4000002)),
                              "O_value"_(createIntSpanOf(1, 2, 3, 4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5))),
                                              "Partition"_("L_key"_("List"_(4)),"Indexes"_(4)),
                                              "Partition"_("L_key"_("List"_(5)),"Indexes"_(3)),
                                              "Partition"_("L_key"_("List"_(20009)),"Indexes"_(2)),
                                              "Partition"_("L_key"_("List"_(4000002)),"Indexes"_(1))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4))),
                                              "Partition"_("O_key"_("List"_(4)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(5)),"Indexes"_(0)),
                                              "Partition"_("O_key"_("List"_(20009)),"Indexes"_(2)),
                                              "Partition"_("O_key"_("List"_(4000002)),"Indexes"_(3))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }
}

TEST_CASE("Join - multi-span", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Simple join 1") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansInt(1,100)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansInt(10000,1)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(2))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(2))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 2") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansInt(100,200)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansInt(1,5)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6)))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6)))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 3") {
    using SpanArguments = boss::expressions::ExpressionSpanArguments;
    std::vector<intType> v1 = {2, 3};
    std::vector<intType> v2 = {100, 101, 102};
    auto s1 = boss::Span<intType>(std::move(v1));
    auto s2 = boss::Span<intType>(std::move(v2));
    SpanArguments args;
    args.emplace_back(std::move(s1));
    args.emplace_back(std::move(s2));
    auto listExpr = boss::expressions::ComplexExpression("List"_, {}, {}, std::move(args));

    auto intTable1 = "Table"_("L_key"_(std::move(listExpr)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansInt(10000, 1)),
                              "O_value"_(createTwoSpansInt(1, 4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(1))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(2))),
                            "Where"_("Equal"_("L_key"_,"O_key"_))));
  }
}

TEST_CASE("Join - multi-span-test", "[hazard-adaptive-engine]") {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  SECTION("Simple join 1") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansInt(1,100)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansInt(10000,1)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(2))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(2))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 2") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansIntDecrease(1,100)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansIntDecrease(10000,1)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(2)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(0))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(2)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(0))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 3") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansIntDecrease(1,100)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createIntSpanOf(10002,10001,10000,3,2,1)),
                              "O_value"_(createIntSpanOf(1,2,3,4,5,6))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(2)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(0))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(5)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(4)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(3))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 4") {
    auto intTable1 = "Table"_("L_key"_(createFourSpansIntFrom(1)),
                              "L_value"_(createFourSpansIntFrom(1))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansIntDecrease(10000,1)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(8,4,7,3,6,2,5,1))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(1))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(2)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(0))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 5") {
    auto intTable1 = "Table"_("L_key"_(createFourSpansIntFrom(1)),
                              "L_value"_(createFourSpansIntFrom(1))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createFourSpansIntFrom(4)),
                              "O_value"_(createFourSpansIntFrom(4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(8,4,7,3,6,2,5,1))),
                                              "Partition"_("L_key"_("List"_(4)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(5)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(6)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(7)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(8)),"Indexes"_(0))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(11,7,10,6,9,5,8,4))),
                                              "Partition"_("O_key"_("List"_(4)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(5)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(6)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(7)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(8)),"Indexes"_(0))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }

  SECTION("Simple join 6") {
    auto intTable1 = "Table"_("L_key"_(createTwoSpansInt(1,100)),
                              "L_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto intTable2 = "Table"_("O_key"_(createTwoSpansInt(10000000,1)),
                              "O_value"_(createTwoSpansInt(1,4))); // NOLINT
    auto result = eval("Join"_(std::move(intTable1), std::move(intTable2),
                               "Where"_("Equal"_("L_key"_, "O_key"_))));

    CHECK(result == "Join"_("RadixPartition"_("Table"_("L_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("L_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("L_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("L_key"_("List"_(3)),"Indexes"_(2))),
                            "RadixPartition"_("Table"_("O_value"_("List"_(1,2,3,4,5,6))),
                                              "Partition"_("O_key"_("List"_(1)),"Indexes"_(0)),
                                              "Partition"_("O_key"_("List"_(2)),"Indexes"_(1)),
                                              "Partition"_("O_key"_("List"_(3)),"Indexes"_(2))),
                            "Where"_("Equal"_("L_key"_,"O_key"_)))
    );
  }
}

#if 0
 TEST_CASE("Select multiple predicates with OR", "[hazard-adaptive-engine]") {
   auto engine = boss::engines::BootstrapEngine();
   REQUIRE(!librariesToTest.empty());
   auto eval = [&engine](boss::Expression&& expression) mutable {
     return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                 std::move(expression)));
   };

   SECTION("Selection multiple columns 1") {
     auto intTable =
         "Table"_("Value1"_("List"_(5, 3, 1, 4, 1)), "Value2"_("List"_(1, 2, 3, 4, 5))); // NOLINT
     auto result = eval("Select"_(
         std::move(intTable), "Where"_("Or"_("Greater"_("Value2"_, 3), "Greater"_(3,
         "Value2"_)))));
     CHECK(result == "Table"_("Value1"_("List"_(5, 3, 4, 1)), "Value2"_("List"_(1, 2, 4, 5))));
   }
 }

 To complete - Grouping on multiple columns

  SECTION("Q1 (No Order, No Strings)") {
    auto output = eval("Project"_(
        "Group"_(
            "Project"_(
                "Project"_(
                    "Select"_(
                        "Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                   "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_,
                                         "L_LINESTATUS_INT"_, "L_LINESTATUS_INT"_,
                                         "L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                         "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                         "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_TAX"_,
                                         "L_TAX"_)),
                        "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                    "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
                          "L_LINESTATUS_INT"_, "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                          "L_EXTENDEDPRICE"_, "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                          "Plus"_("L_TAX"_, 1.0), "L_DISCOUNT"_, "L_DISCOUNT"_)),
                "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
                      "L_LINESTATUS_INT"_, "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                      "L_EXTENDEDPRICE"_, "disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_),
                      "charge"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_),
                      "L_DISCOUNT"_, "L_DISCOUNT"_)),
            "By"_("L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_),
            "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
            "Sum"_("L_EXTENDEDPRICE"_),
                  "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_), "SUM_CHARGES"_,
                  "Sum"_("Times"_("DISC_PRICE"_, "calc"_)), "SUM_DISC"_, "Sum"_("L_DISCOUNT"_),
                  "COUNT_ORDER"_, "Count"_)),
        "As"_("L_RETURNFLAG_INT"_, "L_RETURNFLAG_INT"_, "L_LINESTATUS_INT"_,
        "L_LINESTATUS_INT"_,
              "SUM_QTY"_, "SUM_QTY"_, "SUM_BASE_PRICE"_, "SUM_BASE_PRICE"_, "SUM_DISC_PRICE"_,
              "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
              "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
              "Divide"_("SUM_BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
              "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)));
    CHECK(output ==
          "Table"_("L_RETURNFLAG_INT"_("List"_('N'_i64, 'A'_i64)),                      //
          NOLINT
                   "L_LINESTATUS_INT"_("List"_('O'_i64, 'F'_i64)),                      //
                   NOLINT "SUM_QTY"_("List"_(17 + 21, 8 + 5)), // NOLINT
                   "SUM_BASE_PRICE"_("List"_(17954.55 + 34850.16, 7712.48 + 25284.00)), //
                   NOLINT "SUM_DISC_PRICE"_(
                       "List"_(17954.55 * (1.0 - 0.10) + 34850.16 * (1.0 - 0.05),       //
                       NOLINT
                               7712.48 * (1.0 - 0.06) + 25284.00 * (1.0 - 0.06))),      //
                               NOLINT
                   "SUM_CHARGES"_("List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0) +      //
                   NOLINT
                                              34850.16 * (1.0 - 0.05) * (0.06 + 1.0),   //
                                              NOLINT
                                          7712.48 * (1.0 - 0.06) * (0.02 + 1.0) +       //
                                          NOLINT
                                              25284.00 * (1.0 - 0.06) * (0.06 + 1.0))), //
                                              NOLINT
                   "AVG_PRICE"_("List"_((17954.55 + 34850.16) / 2,                      //
                   NOLINT
                                        (7712.48 + 25284.00) / 2)),                     //
                                        NOLINT
                   "AVG_DISC"_("List"_((0.10 + 0.05) / 2, (0.06 + 0.06) / 2)),          //
                   NOLINT "COUNT_ORDER"_("List"_(2, 2)))); // NOLINT
  }

  SECTION("Q1 (No Order)") {
    auto output = eval("Project"_(
        "Group"_(
            "Project"_(
                "Project"_(
                    "Select"_("Project"_(lineitem.clone(CloneReason::FOR_TESTING),
                                         "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                         "L_LINESTATUS"_,
                                               "L_LINESTATUS"_, "L_QUANTITY"_, "L_QUANTITY"_,
                                               "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                                               "L_SHIPDATE"_, "L_EXTENDEDPRICE"_,
                                               "L_EXTENDEDPRICE"_, "L_TAX"_, "L_TAX"_)),
                              "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                    "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                          "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                          "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                          "Plus"_("L_TAX"_, 1.0), "L_DISCOUNT"_, "L_DISCOUNT"_)),
                "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_,
                      "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                      "disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                      "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                      "L_DISCOUNT"_)),
            "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_),
            "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
            "Sum"_("L_EXTENDEDPRICE"_),
                  "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_), "SUM_CHARGES"_,
                  "Sum"_("Times"_("DISC_PRICE"_, "calc"_)), "SUM_DISC"_, "Sum"_("L_DISCOUNT"_),
                  "COUNT_ORDER"_, "Count"_)),
        "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_, "SUM_QTY"_,
              "SUM_QTY"_, "SUM_BASE_PRICE"_, "SUM_BASE_PRICE"_, "SUM_DISC_PRICE"_,
              "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
              "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
              "Divide"_("SUM_BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
              "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)));
    CHECK(output ==
          "Table"_("L_RETURNFLAG"_("List"_("N", "A")),                                  //
          NOLINT
                   "L_LINESTATUS"_("List"_("O", "F")),                                  //
                   NOLINT "SUM_QTY"_("List"_(17 + 21, 8 + 5)), // NOLINT
                   "SUM_BASE_PRICE"_("List"_(17954.55 + 34850.16, 7712.48 + 25284.00)), //
                   NOLINT "SUM_DISC_PRICE"_(
                       "List"_(17954.55 * (1.0 - 0.10) + 34850.16 * (1.0 - 0.05),       //
                       NOLINT
                               7712.48 * (1.0 - 0.06) + 25284.00 * (1.0 - 0.06))),      //
                               NOLINT
                   "SUM_CHARGES"_("List"_(17954.55 * (1.0 - 0.10) * (0.02 + 1.0) +      //
                   NOLINT
                                              34850.16 * (1.0 - 0.05) * (0.06 + 1.0),   //
                                              NOLINT
                                          7712.48 * (1.0 - 0.06) * (0.02 + 1.0) +       //
                                          NOLINT
                                              25284.00 * (1.0 - 0.06) * (0.06 + 1.0))), //
                                              NOLINT
                   "AVG_PRICE"_("List"_((17954.55 + 34850.16) / 2,                      //
                   NOLINT
                                        (7712.48 + 25284.00) / 2)),                     //
                                        NOLINT
                   "AVG_DISC"_("List"_((0.10 + 0.05) / 2, (0.06 + 0.06) / 2)),          //
                   NOLINT "COUNT_ORDER"_("List"_(2, 2)))); // NOLINT
  }
#endif

int main(int argc, char* argv[]) {
  Catch::Session session;
  session.configData().showSuccessfulTests = true;
  session.cli(session.cli() | Catch::clara::Opt(librariesToTest, "library")["--library"]);
  auto const returnCode = session.applyCommandLine(argc, argv);
  if(returnCode != 0) {
    return returnCode;
  }
  return session.run();
}
// NOLINTEND(readability-function-cognitive-complexity)
// NOLINTEND(bugprone-exception-escape)
// NOLINTEND(readability-magic-numbers)
