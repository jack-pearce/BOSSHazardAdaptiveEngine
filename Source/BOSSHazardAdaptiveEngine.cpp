#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>
#include <Algorithm.hpp>
#include <stdexcept>
#include <variant>
#include <iostream>

#include <any>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "CoreCandidates.hpp"

using std::string_literals::operator""s;
using boss::utilities::operator""_;
using boss::expressions::generic::isComplexExpression;
using boss::utilities::overload;

using namespace boss::algorithm;
using namespace boss;


static Expression evaluate(Expression&& expression);

class Pred : public std::function<std::optional<Span<Pred>>(ExpressionArguments&, bool, bool)> {
public:
  using Function = std::function<std::optional<Span<Pred>>(ExpressionArguments&, bool, bool)>;
  template <typename F>
  Pred(F&& func, boss::Expression&& expr)
      : Function(std::forward<F>(func)), cachedExpr(std::move(expr)) {}
  template <typename F>
  Pred(F&& func, boss::Symbol const& s) : Function(std::forward<F>(func)), cachedExpr(s) {}
  Pred(Pred&& other) noexcept
      : Function(std::move(static_cast<Function&&>(other))),
        cachedExpr(std::move(other.cachedExpr)) {}
  Pred& operator=(Pred&& other) noexcept {
    *static_cast<Function*>(this) = std::move(static_cast<Function&&>(other));
    cachedExpr = std::move(other.cachedExpr);
    return *this;
  }
  Pred(Pred const&) = delete;
  Pred const& operator=(Pred const&) = delete;
  ~Pred() = default;

  friend ::std::ostream& operator<<(std::ostream& out, Pred const& pred) {
    out << "[Pred for " << pred.cachedExpr << "]";
    return out;
  }

  explicit operator boss::Expression() && { return std::move(cachedExpr); }

private:
  boss::Expression cachedExpr; // so we can revert it back if unused
};

template <typename... Args> class TypedFunctor;

class Functor {
public:
  virtual ~Functor() = default;
  Functor() = default;
  Functor(Functor const&) = delete;
  Functor const& operator=(Functor const&) = delete;
  Functor(Functor&&) = delete;
  Functor const& operator=(Functor&&) = delete;
  virtual std::pair<Expression, bool> operator()(ComplexExpression&& e) = 0;
  template <typename Func> static std::unique_ptr<Functor> makeUnique(Func&& func) {
    return std::unique_ptr<Functor>(new TypedFunctor(std::forward<decltype(func)>(func)));
  }
};

template <typename... Args> class TypedFunctor : public Functor {
public:
  ~TypedFunctor() override = default;
  TypedFunctor(TypedFunctor const&) = delete;
  TypedFunctor const& operator=(TypedFunctor const&) = delete;
  TypedFunctor(TypedFunctor&&) = delete;
  TypedFunctor const& operator=(TypedFunctor&&) = delete;
  explicit TypedFunctor(
      std::function<Expression(ComplexExpressionWithStaticArguments<Args...>&&)> f)
      : func(f) {}
  std::pair<Expression, bool> operator()(ComplexExpression&& e) override {
    return dispatchAndEvaluate(std::move(e));
  }

private:
  std::function<Expression(ComplexExpressionWithStaticArguments<Args...>&&)> func;
  template <typename... T>
  std::pair<Expression, bool> dispatchAndEvaluate(ComplexExpressionWithStaticArguments<T...>&& e) {
    auto [head, statics, dynamics, spans] = std::move(e).decompose();
    if constexpr(sizeof...(T) < sizeof...(Args)) {
      Expression dispatchArgument =
          dynamics.empty()
              ? std::visit(
                    [](auto& a) -> Expression {
                      if constexpr(std::is_same_v<std::decay_t<decltype(a)>, Span<Pred const>>) {
                        throw std::runtime_error(
                            "Found a Span<Pred const> in an expression to evaluate. "
                            "It should not happen.");
                      } else if constexpr(std::is_same_v<std::decay_t<decltype(a)>, Span<bool>>) {
                        return bool(a[0]);
                      } else {
                        return std::move(a[0]);
                      }
                    },
                    spans.front())
              : std::move(dynamics.at(sizeof...(T)));
      if(dynamics.empty()) {
        spans[0] = std::visit(
            [](auto&& span) -> boss::expressions::ExpressionSpanArgument {
              return std::forward<decltype(span)>(span).subspan(1);
            },
            std::move(spans[0]));
      }
      return std::visit(
          [head = std::move(head), statics = std::move(statics), dynamics = std::move(dynamics),
           spans = std::move(spans), this](auto&& argument) mutable -> std::pair<Expression, bool> {
            typedef std::decay_t<decltype(argument)> ArgType;

            if constexpr(std::is_same_v<ArgType,
                                        std::tuple_element_t<sizeof...(T), std::tuple<Args...>>>) {
              // argument type matching, add one more static argument to the expression
              return dispatchAndEvaluate(ComplexExpressionWithStaticArguments<T..., ArgType>(
                  head,
                  std::tuple_cat(std::move(statics),
                                 std::make_tuple(std::forward<decltype(argument)>(argument))),
                  std::move(dynamics), std::move(spans)));
            } else {
              ExpressionArguments rest{};
              rest.emplace_back(std::forward<decltype(argument)>(argument));
              if(dynamics.size() > sizeof...(Args)) {
                std::copy(std::move_iterator(next(dynamics.begin(), sizeof...(Args))),
                          std::move_iterator(dynamics.end()), std::back_inserter(rest));
              }
              // failed to match the arguments, return the non/semi-evaluated expression
              return std::make_pair(
                  ComplexExpressionWithStaticArguments<T...>(std::move(head), std::move(statics),
                                                             std::move(rest), std::move(spans)),
                  false);
            }
          },
          evaluate(std::move(dispatchArgument)));
    } else {
      ExpressionArguments rest{};
      if(dynamics.size() > sizeof...(Args)) {
        std::transform(std::move_iterator(next(dynamics.begin(), sizeof...(Args))),
                       std::move_iterator(dynamics.end()), std::back_inserter(rest),
                       [](auto&& arg) {
                         // evaluate the rest of the arguments
                         return evaluate(std::forward<decltype(arg)>(arg));
                       });
      }
      // all the arguments are matching, call the function and return the evaluated expression
      return std::make_pair(
          func(ComplexExpressionWithStaticArguments<Args...>(std::move(head), std::move(statics),
                                                             std::move(rest), std::move(spans))),
          true);
    }
  }
};

// from https://stackoverflow.com/a/7943765
template <typename F> struct function_traits : public function_traits<decltype(&F::operator())> {};
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const> {
  typedef ReturnType result_type;
  typedef std::tuple<Args...> args;
};

template <typename F>
concept supportsFunctionTraits =
    requires(F&& f) { typename function_traits<decltype(&F::operator())>; };

class StatelessOperator {
public:
  std::map<std::type_index, std::unique_ptr<Functor>> functors;

  Expression operator()(ComplexExpression&& e) {
    for(auto&& [id, f] : functors) {
      auto [output, success] = (*f)(std::move(e));
      if(success) {
        return std::move(output);
      }
      e = std::get<ComplexExpression>(std::move(output));
    }
    return std::move(e);
  }

  template <typename F, typename... Types>
  void registerFunctorForTypes(F f, std::variant<Types...> unused) {
    (
      [this, &f]() {
        if constexpr(std::is_invocable_v<F, ComplexExpressionWithStaticArguments<Types>>) {
          this->functors[typeid(Types)] = Functor::makeUnique(
              std::function<Expression(ComplexExpressionWithStaticArguments<Types>&&)>(f));
        }
      }(),
      ...);
  }

  template <typename F>
  StatelessOperator& operator=(F&& f)
      requires(!supportsFunctionTraits<F>)
  {
    registerFunctorForTypes(f, Expression{});
    if constexpr(std::is_invocable_v<F, ComplexExpression>) {
      this->functors[typeid(ComplexExpression)] =
          Functor::makeUnique(std::function<Expression(ComplexExpression&&)>(f));
    }
    return *this;
  }

  template <typename F>
  StatelessOperator& operator=(F&& f)
      requires supportsFunctionTraits<F>
  {
    using FuncInfo = function_traits<F>;
    using ComplexExpressionArgs = std::tuple_element_t<0, typename FuncInfo::args>;
    this->functors[typeid(ComplexExpressionArgs ::StaticArgumentTypes)] =
        Functor::makeUnique(std::function<Expression(ComplexExpressionArgs&&)>(std::forward<F>(f)));
    return *this;
  }
};

template <typename T>
concept NumericType = requires(T param) {
  requires std::is_integral_v<T> || std::is_floating_point_v<T>;
  requires !std::is_same_v<bool, T>;
  requires std::is_arithmetic_v<decltype(param + 1)>;
  requires !std::is_pointer_v<T>;
};

class OperatorMap : public std::unordered_map<boss::Symbol, StatelessOperator> {
public:
  OperatorMap() {
    (*this)["Plus"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      ExpressionArguments args = input.getArguments();
      return visitAccumulate(std::move(args), 0L, [](auto&& state, auto&& arg) {
        if constexpr(std::is_integral_v<std::decay_t<decltype(arg)>>) {
          state += arg;
        }
        return state;
      });
    };

    (*this)["Multiply"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      ExpressionArguments args = input.getArguments();
      return visitAccumulate(std::move(args), 1L, [](auto&& state, auto&& arg) {
        if constexpr(std::is_integral_v<std::decay_t<decltype(arg)>>) {
          state *= arg;
        }
        return state;
      });
    };
  }
};

static Expression evaluate(Expression&& expression) {
  static OperatorMap operators;
  return std::move(expression) | [](ComplexExpression&& e) -> Expression {
    auto head = e.getHead();
    auto it = operators.find(head);
    if(it != operators.end()) {
      return it->second(std::move(e));
    } else {
      auto [_, unused_, dynamics, spans] = std::move(e).decompose();
      return ComplexExpression{std::move(head), {}, std::move(dynamics), std::move(spans)};
    }
  };
}

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
}

// As close to array fire engine as possible
/*static Expression evaluate(Expression&& expression) {
  static OperatorMap operators;
  return visit(
      boss::utilities::overload(
          [](ComplexExpression&& e) -> Expression {
            auto head = e.getHead();
            auto it = operators.find(head);
            if(it != operators.end()) {
              return it->second(std::move(e));
            }
            // at least evaluate all the arguments
            auto [_, unused_, dynamics, spans] = std::move(e).decompose();
            std::transform(
                std::make_move_iterator(dynamics.begin()), std::make_move_iterator(dynamics.end()),
                dynamics.begin(),
                [](auto&& arg) { return evaluate(std::forward<decltype(arg)>(arg)); });
            return ComplexExpression{std::move(head), {}, std::move(dynamics), std::move(spans)};
          },
          [](auto&& e) -> Expression { return std::forward<decltype(e)>(e); }),
      std::forward<decltype(expression)>(expression));
}*/

// Basic engine example using operator| utility function
/*static Expression evaluate(Expression&& expression) {
  return std::move(expression) | [](ComplexExpression&& e) -> Expression {
    ExpressionArguments args = e.getArguments();
    visitTransform(args, [](auto&& arg) -> Expression {
      if constexpr(isComplexExpression<decltype(arg)>) {
        return evaluate(std::move(arg));
      } else {
        return std::forward<decltype(arg)>(arg);
      }
    });

    if (e.getHead() == Symbol("Plus")) {
      return visitAccumulate(std::move(args), 0L, [](auto&& state, auto&& arg) {
        if constexpr(std::is_integral_v<std::decay_t<decltype(arg)>>) {
          state += arg;
        }
        return state;
      });

    } else if (e.getHead() == Symbol("Multiply")) {
      return visitAccumulate(std::move(args), 1L, [](auto&& state, auto&& arg) {
        if constexpr(std::is_integral_v<std::decay_t<decltype(arg)>>) {
          state *= arg;
        }
        return state;
      });

    } else {
      return ComplexExpression(e.getHead(), {}, std::move(args), {});
    }
  };
}*/

// Basic engine example not using operator| utility function
/*static Expression evaluate(Expression&& expression) {
  return std::visit(
    [](auto&& e) -> Expression {
    if constexpr(isComplexExpression<decltype(e)>) {
      ExpressionArguments args = e.getArguments();
      visitTransform(args, [](auto&& arg) -> Expression {
        if constexpr(isComplexExpression<decltype(arg)>) {
          return evaluate(std::move(arg));
        } else {
          return std::forward<decltype(arg)>(arg);
        }
      });
      if (e.getHead() == Symbol("Plus")) {
        return visitAccumulate(std::move(args), 0L, [](auto&& state, auto&& arg) {
          if constexpr(std::is_integral_v<std::decay_t<decltype(arg)>>) {
            state += arg;
          }
          return state;
        });
      } else {
        return ComplexExpression(e.getHead(), {}, std::move(args), {});
      }
    } else {
      return std::forward<decltype(e)>(e);
    }
  },
  std::move(expression));
}*/
