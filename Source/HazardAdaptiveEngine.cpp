#include <Algorithm.hpp>
#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>

#include "config.hpp"
#include "operators/select.hpp"
#include "utilities/memory.hpp"

#include <algorithm>
#include <any>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

class Pred;

using HAExpressionSystem = boss::expressions::generic::ExtensibleExpressionSystem<Pred, uint32_t>;
using AtomicExpression = HAExpressionSystem::AtomicExpression;
using ComplexExpression = HAExpressionSystem::ComplexExpression;
template <typename... T>
using ComplexExpressionWithStaticArguments =
    HAExpressionSystem::ComplexExpressionWithStaticArguments<T...>;
using Expression = HAExpressionSystem::Expression;
using ExpressionArguments = HAExpressionSystem::ExpressionArguments;
using ExpressionSpanArguments = HAExpressionSystem::ExpressionSpanArguments;
using ExpressionSpanArgument = HAExpressionSystem::ExpressionSpanArgument;
using ExpressionBuilder = boss::utilities::ExtensibleExpressionBuilder<HAExpressionSystem>;
static ExpressionBuilder operator""_(const char* name, size_t /*unused*/) {
  return ExpressionBuilder(name);
}

using BOSSExpressionSpanArguments = boss::DefaultExpressionSystem::ExpressionSpanArguments;
using BOSSExpressionSpanArgument = boss::DefaultExpressionSystem::ExpressionSpanArgument;

using adaptive::config::DOP;
using adaptive::config::selectImplementation;

using boss::Span;
using boss::Symbol;
using SpanInputs = std::variant<std::vector<std::int64_t>, std::vector<std::double_t>,
                                std::vector<std::string>, std::vector<uint32_t>>;

using namespace boss::algorithm;

/** Pred function takes a relation in the form of ExpressionArguments, and an optional pointer to
 * a span of candidate indexes. Based on these inputs and the internal predicate it returns an
 * optional span in the form of an ExpressionSpanArgument.
 */
class Pred : public std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&,
                                                                        Span<uint32_t>*)> {
public:
  using Function =
      std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&, Span<uint32_t>*)>;
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
  boss::Expression cachedExpr;
};

#ifdef DEBUG_MODE
namespace utilities {
static boss::Expression injectDebugInfoToSpans(boss::Expression&& expr) {
  return std::visit(
      boss::utilities::overload(
          [&](boss::ComplexExpression&& e) -> boss::Expression {
            auto [head, unused_, dynamics, spans] = std::move(e).decompose();
            boss::ExpressionArguments debugDynamics;
            debugDynamics.reserve(dynamics.size() + spans.size());
            std::transform(std::make_move_iterator(dynamics.begin()),
                           std::make_move_iterator(dynamics.end()),
                           std::back_inserter(debugDynamics), [](auto&& arg) {
                             return injectDebugInfoToSpans(std::forward<decltype(arg)>(arg));
                           });
            std::transform(
                std::make_move_iterator(spans.begin()), std::make_move_iterator(spans.end()),
                std::back_inserter(debugDynamics), [](auto&& span) {
                  return std::visit(
                      [](auto&& typedSpan) -> boss::Expression {
                        using Element = typename std::decay_t<decltype(typedSpan)>::element_type;
                        return boss::ComplexExpressionWithStaticArguments<std::string, int64_t>(
                            "Span"_, {typeid(Element).name(), typedSpan.size()}, {}, {});
                      },
                      std::forward<decltype(span)>(span));
                });
            return boss::ComplexExpression(std::move(head), {}, std::move(debugDynamics), {});
          },
          [](auto&& otherTypes) -> boss::Expression { return otherTypes; }),
      std::move(expr));
}

static Expression injectDebugInfoToSpansExtendedExpressionSystem(Expression&& expr) {
  return std::visit(
      boss::utilities::overload(
          [&](ComplexExpression&& e) -> Expression {
            auto [head, unused_, dynamics, spans] = std::move(e).decompose();
            ExpressionArguments debugDynamics;
            debugDynamics.reserve(dynamics.size() + spans.size());
            std::transform(std::make_move_iterator(dynamics.begin()),
                           std::make_move_iterator(dynamics.end()),
                           std::back_inserter(debugDynamics), [](auto&& arg) {
                             return injectDebugInfoToSpansExtendedExpressionSystem(
                                 std::forward<decltype(arg)>(arg));
                           });
            std::transform(
                std::make_move_iterator(spans.begin()), std::make_move_iterator(spans.end()),
                std::back_inserter(debugDynamics), [](auto&& span) {
                  return std::visit(
                      [](auto&& typedSpan) -> Expression {
                        using Element = typename std::decay_t<decltype(typedSpan)>::element_type;
                        return ComplexExpressionWithStaticArguments<std::string, int64_t>(
                            "Span"_, {typeid(Element).name(), typedSpan.size()}, {}, {});
                      },
                      std::forward<decltype(span)>(span));
                });
            return ComplexExpression(std::move(head), {}, std::move(debugDynamics), {});
          },
          [](auto&& otherTypes) -> Expression { return otherTypes; }),
      std::move(expr));
}
} // namespace utilities
#endif

static Expression evaluateInternal(Expression&& e);

// Q: Should tests all use spans for data - currently most using dynamics which requires this func
template <typename... StaticArgumentTypes>
ComplexExpressionWithStaticArguments<StaticArgumentTypes...>
transformDynamicsToSpans(ComplexExpressionWithStaticArguments<StaticArgumentTypes...>&& input_) {
  std::vector<SpanInputs> spanInputs;
  auto [head, statics, dynamics, oldSpans] = std::move(input_).decompose();
  for(auto it = std::move_iterator(dynamics.begin()); it != std::move_iterator(dynamics.end());
      ++it) {
    std::visit(
        [&spanInputs]<typename InputType>(InputType&& argument) {
          using Type = std::decay_t<InputType>;
          if constexpr(boss::utilities::isVariantMember<std::vector<Type>, SpanInputs>::value) {
            if(!spanInputs.empty() &&
               std::holds_alternative<std::vector<Type>>(spanInputs.back())) {
              std::get<std::vector<Type>>(spanInputs.back()).push_back(argument);
            } else {
              spanInputs.push_back(std::vector<Type>{argument});
            }
          }
        },
        evaluateInternal(*it));
  }
  dynamics.erase(dynamics.begin(), dynamics.end());
  ExpressionSpanArguments spans;
  std::transform(std::move_iterator(spanInputs.begin()), std::move_iterator(spanInputs.end()),
                 std::back_inserter(spans), [](auto&& untypedInput) {
                   return std::visit(
                       []<typename Element>(std::vector<Element>&& input)
                           -> ExpressionSpanArgument { return Span<Element>(std::move(input)); },
                       std::move(untypedInput));
                 });

  std::copy(std::move_iterator(oldSpans.begin()), std::move_iterator(oldSpans.end()),
            std::back_inserter(spans));
  return {head, std::move(statics), std::move(dynamics), std::move(spans)};
}

Expression transformDynamicsToSpans(Expression&& input) {
  return std::visit(
      [](auto&& x) -> Expression {
        if constexpr(std::is_same_v<std::decay_t<decltype(x)>, ComplexExpression>)
          return transformDynamicsToSpans(std::move(x));
        else
          return x;
      },
      std::move(input));
}

ExpressionArguments transformDynamicsInColumnsToSpans(ExpressionArguments&& columns) {
  std::transform(
      std::make_move_iterator(columns.begin()), std::make_move_iterator(columns.end()),
      columns.begin(), [](auto&& columnExpr) {
        auto column = get<ComplexExpression>(std::forward<decltype(columnExpr)>(columnExpr));
        auto [head, unused_, dynamics, spans] = std::move(column).decompose();
        auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
        if(list.getDynamicArguments().size() > 0) {
          list = transformDynamicsToSpans(std::move(list));
        }
        dynamics.at(0) = std::move(list);
        return ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
      });
  return std::move(columns);
}

static boss::Expression toBOSSExpression(Expression&& expr) {
  return std::visit(
      boss::utilities::overload(
          [&](ComplexExpression&& e) -> boss::Expression {
            auto [head, unused_, dynamics, spans] = std::move(e).decompose();
            boss::ExpressionArguments bossDynamics;
            bossDynamics.reserve(dynamics.size());
            std::transform(
                std::make_move_iterator(dynamics.begin()), std::make_move_iterator(dynamics.end()),
                std::back_inserter(bossDynamics),
                [](auto&& arg) { return toBOSSExpression(std::forward<decltype(arg)>(arg)); });
            BOSSExpressionSpanArguments bossSpans;
            std::transform(
                std::make_move_iterator(spans.begin()), std::make_move_iterator(spans.end()),
                std::back_inserter(bossSpans), [](auto&& span) {
                  return std::visit(
                      []<typename T>(Span<T>&& typedSpan) -> BOSSExpressionSpanArgument {
                        if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t> ||
                                     std::is_same_v<T, std::string>) {
                          return typedSpan;
                        } else {
                          throw std::runtime_error("span type not supported by BOSS core");
                        }
                      },
                      std::forward<decltype(span)>(span));
                });
            return boss::ComplexExpression(std::move(head), {}, std::move(bossDynamics),
                                           std::move(bossSpans));
          },
          [&](Pred&& e) -> boss::Expression {
            boss::Expression output = static_cast<boss::Expression>(std::move(e));
            return output;
          },
          [](auto&& otherTypes) -> boss::Expression { return otherTypes; }),
      std::move(expr));
}

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
    auto [head1, statics1, dynamics1, spans1] = std::move(e).decompose();
    if constexpr(sizeof...(T) < sizeof...(Args)) {
      Expression dispatchArgument =
          dynamics1.empty()
              ? std::visit(
                    [](auto& a) -> Expression {
                      if constexpr(std::is_same_v<std::decay_t<decltype(a)>, Span<Pred const>>) {
                        throw std::runtime_error(
                            "Found a Span<Pred const> in an expression to evaluate. "
                            "It should not happen.");
                      } else if constexpr(std::is_same_v<std::decay_t<decltype(a)>, Span<bool>> ||
                                          std::is_same_v<std::decay_t<decltype(a)>,
                                                         Span<bool const>>) {
                        return bool(a[0]);
                      } else {
                        return std::move(a[0]);
                      }
                    },
                    spans1.front())
              : std::move(dynamics1.at(sizeof...(T)));
      if(dynamics1.empty()) {
        spans1[0] = std::visit(
            [](auto&& span) -> ExpressionSpanArgument {
              return std::forward<decltype(span)>(span).subspan(1);
            },
            std::move(spans1[0]));
      }

      return std::visit(
          [head = std::move(head1), statics = std::move(statics1), dynamics = std::move(dynamics1),
           spans = std::move(spans1),
           this](auto&& argument) mutable -> std::pair<Expression, bool> {
            typedef std::decay_t<decltype(argument)> ArgType;

            if constexpr(std::is_same_v<ArgType, typename std::tuple_element<
                                                     sizeof...(T), std::tuple<Args...>>::type>) {
              // argument type matching, add one more static argument to the expression
              return this->dispatchAndEvaluate(ComplexExpressionWithStaticArguments<T..., ArgType>(
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
                  ComplexExpressionWithStaticArguments<T...>(head, std::move(statics),
                                                             std::move(rest), std::move(spans)),
                  false);
            }
          },
          evaluateInternal(std::move(dispatchArgument)));
    } else {
      ExpressionArguments rest{};
      if(dynamics1.size() > sizeof...(Args)) {
        std::transform(
            std::move_iterator(next(dynamics1.begin(), sizeof...(Args))),
            std::move_iterator(dynamics1.end()), std::back_inserter(rest),
            [](auto&& arg) { return evaluateInternal(std::forward<decltype(arg)>(arg)); });
      }
      return std::make_pair(
          func(ComplexExpressionWithStaticArguments<Args...>(std::move(head1), std::move(statics1),
                                                             std::move(rest), std::move(spans1))),
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

template <typename F> concept supportsFunctionTraits = requires(F && f) {
  typename function_traits<decltype(&F::operator())>;
};

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
  void registerFunctorForTypes(F f, std::variant<Types...> /*unused*/) {
    (
        [this, &f]() {
          if constexpr(std::is_invocable_v<F, ComplexExpressionWithStaticArguments<Types>>) {
            this->functors[typeid(Types)] = Functor::makeUnique(
                std::function<Expression(ComplexExpressionWithStaticArguments<Types>&&)>(f));
          }
          (void)f;    /* Addresses this bug https://bugs.llvm.org/show_bug.cgi?id=35450 */
          (void)this; /* Addresses this bug https://bugs.llvm.org/show_bug.cgi?id=35450 */
        }(),
        ...);
  }

  template <typename F> StatelessOperator& operator=(F&& f) requires(!supportsFunctionTraits<F>) {
    registerFunctorForTypes(f, Expression{});
    if constexpr(std::is_invocable_v<F, ComplexExpression>) {
      this->functors[typeid(ComplexExpression)] =
          Functor::makeUnique(std::function<Expression(ComplexExpression&&)>(f));
    }
    return *this;
  }

  template <typename F> StatelessOperator& operator=(F&& f) requires supportsFunctionTraits<F> {
    using FuncInfo = function_traits<F>;
    using ComplexExpressionArgs = std::tuple_element_t<0, typename FuncInfo::args>;
    this->functors[typeid(ComplexExpressionArgs /*::StaticArgumentTypes*/)] =
        Functor::makeUnique(std::function<Expression(ComplexExpressionArgs&&)>(std::forward<F>(f)));
    return *this;
  }
};

template <typename T> concept NumericType = requires(T param) {
  requires std::is_integral_v<T> || std::is_floating_point_v<T>;
  requires !std::is_same_v<bool, T>;
  requires std::is_arithmetic_v<decltype(param + 1)>;
  requires !std::is_pointer_v<T>;
};

class OperatorMap : public std::unordered_map<boss::Symbol, StatelessOperator> {
private:
  template <typename T1, typename T2, typename F>
  static ExpressionSpanArgument createLambdaResult(T1&& arg1, T2&& arg2, F& f) {
    ExpressionSpanArgument span;
    std::visit(
        [&span, &f](auto&& typedSpan1, auto&& typedSpan2) {
          using Type1 = std::decay_t<decltype(typedSpan1)>;
          using Type2 = std::decay_t<decltype(typedSpan2)>;
          if constexpr((std::is_same_v<Type1, Span<int64_t>> &&
                        std::is_same_v<Type2, Span<int64_t>>) ||
                       (std::is_same_v<Type1, Span<double>> &&
                        std::is_same_v<Type2, Span<double>>)) {
            using ElementType1 = typename Type1::element_type;
            using ElementType2 = typename Type2::element_type;
            using OutputType = decltype(std::declval<decltype(f)>()(std::declval<ElementType1>(),
                                                                    std::declval<ElementType2>()));
            // If output of f is a bool, then we are performing a select so return indexes
            if constexpr(std::is_same_v<OutputType, bool>) {
              assert(typedSpan1.size() == 1 || typedSpan2.size() == 1);
              if(typedSpan2.size() == 1) {
                span = adaptive::select(selectImplementation, typedSpan1, typedSpan2[0], true, f,
                                        {}, DOP);
              } else {
                span = adaptive::select(selectImplementation, typedSpan2, typedSpan1[0], false, f,
                                        {}, DOP);
              }
            } else {
              std::vector<OutputType> output;
              output.reserve(std::max(typedSpan1.size(), typedSpan2.size()));
              if(typedSpan2.size() == 1) {
                for(size_t i = 0; i < typedSpan1.size(); ++i) {
                  output.push_back(f(typedSpan1[i], typedSpan2[0]));
                }
              } else if(typedSpan1.size() == 1) {
                for(size_t i = 0; i < typedSpan2.size(); ++i) {
                  output.push_back(f(typedSpan1[0], typedSpan2[i]));
                }
              } else {
                assert(typedSpan1.size() == typedSpan2.size());
                for(size_t i = 0; i < typedSpan2.size(); ++i) {
                  output.push_back(f(typedSpan1[i], typedSpan2[i]));
                }
              }
              span = Span<OutputType>(std::move(std::vector(output)));
            }
          } else {
            throw std::runtime_error("unsupported column type in lambda: " +
                                     std::string(typeid(typename Type1::element_type).name()));
          }
        },
        std::forward<T1>(arg1), std::forward<T2>(arg2));
    return span;
  }

  template <typename T1, typename T2, typename F>
  static ExpressionSpanArgument createLambdaPipelineResult(T1&& arg1, T2&& arg2, F& f,
                                                           Span<uint32_t>& indexes) {
    std::visit(
        [&indexes, &f](auto&& typedSpan1, auto&& typedSpan2) {
          using Type1 = std::decay_t<decltype(typedSpan1)>;
          using Type2 = std::decay_t<decltype(typedSpan2)>;
          if constexpr((std::is_same_v<Type1, Span<int64_t>> &&
                        std::is_same_v<Type2, Span<int64_t>>) ||
                       (std::is_same_v<Type1, Span<double>> &&
                        std::is_same_v<Type2, Span<double>>)) {
            using ElementType1 = typename Type1::element_type;
            using ElementType2 = typename Type2::element_type;
            using OutputType = decltype(std::declval<decltype(f)>()(std::declval<ElementType1>(),
                                                                    std::declval<ElementType2>()));

            if constexpr(std::is_same_v<OutputType, bool>) {
              assert(typedSpan1.size() == 1 || typedSpan2.size() == 1);
              if(typedSpan2.size() == 1) {
                indexes = adaptive::select(selectImplementation, typedSpan1, typedSpan2[0], true, f,
                                           std::move(indexes), DOP);
              } else {
                indexes = adaptive::select(selectImplementation, typedSpan2, typedSpan1[0], false,
                                           f, std::move(indexes), DOP);
              }
            } else {
              throw std::runtime_error(
                  "function in createLambdaPipelineResult does not return bool: " +
                  std::string(typeid(typename Type1::element_type).name()));
            }
          } else {
            throw std::runtime_error("unsupported column type in lambda: " +
                                     std::string(typeid(typename Type1::element_type).name()));
          }
        },
        std::forward<T1>(arg1), std::forward<T2>(arg2));
    return {};
  }

  // Q: To create ComplexExpressionWithStaticArguments you must always explicitly call the
  // constructor with the associated types?
  template <typename T, typename F>
  static Pred createLambdaExpression(ComplexExpressionWithStaticArguments<T>&& e, F&& f) {
    assert(e.getSpanArguments().empty());
    assert(e.getDynamicArguments().size() >= 1);
    if(e.getDynamicArguments().size() == 1) {
      Pred::Function pred = std::visit(
          [&e, &f](auto&& arg) -> Pred::Function {
            return [pred1 = createLambdaArgument(get<0>(e.getStaticArguments())),
                    pred2 = createLambdaArgument(arg),
                    &f](ExpressionArguments& columns,
                        Span<uint32_t>* indexes) mutable -> std::optional<ExpressionSpanArgument> {
              auto arg1 = pred1(columns, nullptr);
              auto arg2 = pred2(columns, nullptr);
              if(!arg1 || !arg2) {
                return {};
              }
              if(indexes) {
                return createLambdaPipelineResult(std::move(*arg1), std::move(*arg2), f, *indexes);
              }
              return createLambdaResult(std::move(*arg1), std::move(*arg2), f);
            };
          },
          // Q: Could this be a single funtion call, getDynamicArgumentAt(0)?
          e.getDynamicArguments().at(0));
      return {std::move(pred), toBOSSExpression(std::move(e))};
    } else {
      Pred::Function pred = std::accumulate(
          e.getDynamicArguments().begin(), e.getDynamicArguments().end(),
          (Pred::Function&&)createLambdaArgument(get<0>(e.getStaticArguments())),
          [&f](auto&& acc, auto const& e) -> Pred::Function {
            return std::visit(
                [&f, &acc](auto&& arg) -> Pred::Function {
                  return [acc, pred2 = createLambdaArgument(arg),
                          &f](ExpressionArguments& columns, Span<uint32_t>* indexes) mutable
                         -> std::optional<ExpressionSpanArgument> {
                    assert(!indexes);
                    auto arg1 = acc(columns, nullptr);
                    auto arg2 = pred2(columns, nullptr);
                    if(!arg1 || !arg2) {
                      return {};
                    }
                    return createLambdaResult(std::move(*arg1), std::move(*arg2), f);
                  };
                },
                e);
          });
      return {std::move(pred), toBOSSExpression(std::move(e))};
    }
  }

  template <typename T1, typename T2>
  static Pred createLambdaPipelineOfExpressions(ComplexExpressionWithStaticArguments<T1, T2>&& e) {
    assert(e.getSpanArguments().empty());
    std::vector<Pred::Function> preds;
    preds.reserve(2 + e.getDynamicArguments().size());
    preds.push_back(createLambdaArgument(get<0>(e.getStaticArguments())));
    preds.push_back(createLambdaArgument(get<1>(e.getStaticArguments())));
    for(auto it = e.getDynamicArguments().begin(); it != e.getDynamicArguments().end(); ++it) {
      preds.push_back(get<Pred>(std::move(*it)));
    }
    Pred::Function pred =
        [preds = std::move(preds)](
            ExpressionArguments& columns,
            Span<uint32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      auto candidateIndexes = preds[0](columns, nullptr);
      ExpressionSpanArgument span;
      std::visit(
          [&span, &preds, &columns](auto&& candidateIndexes) {
            if constexpr(std::is_same_v<std::decay_t<decltype(candidateIndexes)>, Span<uint32_t>>) {
              for(auto it = std::next(preds.begin()); it != preds.end(); ++it) {
                if(candidateIndexes.size() == 0) {
                  break;
                }
                (*it)(columns, &candidateIndexes);
              }
              span = Span<uint32_t>(std::move(candidateIndexes));
            } else {
              throw std::runtime_error("Multi-predicate lambda column input is not indexes");
            }
          },
          std::move(*candidateIndexes));
      return span;
    };
    return {std::move(pred), toBOSSExpression(std::move(e))};
  }

  /** Turns single arguments into a single element span so that createLambdaResult acts on two
   *  spans. Could make this cleaner by Pred::Function returning an Expression or variant (but would
   *  require more code)?
   */
  template <typename ArgType> static Pred::Function createLambdaArgument(ArgType const& arg) {
    if constexpr(NumericType<ArgType>) {
      return [arg](ExpressionArguments& /*unused*/,
                   Span<uint32_t>* /*unused*/) -> std::optional<ExpressionSpanArgument> {
        return Span<ArgType>(std::move(std::vector({arg})));
      };
    } else {
      throw std::runtime_error("unsupported argument type in predicate");
      return [](ExpressionArguments& /*unused*/,
                Span<uint32_t>* /*unused*/) -> std::optional<ExpressionSpanArgument> { return {}; };
    }
  }

  static Pred::Function createLambdaArgument(Pred const& arg) {
    return [f = static_cast<Pred::Function const&>(arg)](ExpressionArguments& columns,
                                                         Span<uint32_t>* indexes) {
      return f(columns, indexes);
    };
  }

  /**
   * This function finds a given column and returns a shallow copy of it. Therefore it should
   * only be used locally within an operator and not returned as there is no guarantee the
   * underlying span will stay in scope. For example, it can be called from Select and Group where
   * the columns are used as intermediaries to produce the final result. However, it cannot be used
   * in Project to return columns (instead columns must be moved, see createLambdaArgumentMove func)
   */
  static Pred::Function createLambdaArgument(Symbol const& arg) {
    return [arg](ExpressionArguments& columns,
                 Span<uint32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      for(auto& columnExpr : columns) {
        auto& column = get<ComplexExpression>(columnExpr);
        if(column.getHead().getName() == arg.getName()) {
          // Q: get is a wrapper for std::get? When should get be used?
          auto& span = get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments().at(0);
          return std::visit(
              []<typename T>(Span<T> const& typedSpan) -> std::optional<ExpressionSpanArgument> {
                if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t> ||
                             std::is_same_v<T, std::string>) {
                  using ElementType = std::remove_const_t<T>;
                  auto* ptr = const_cast<ElementType*>(typedSpan.begin());
                  return Span<ElementType>(ptr, typedSpan.size(), []() {});
                } else {
                  throw std::runtime_error("unsupported column type in predicate");
                }
              },
              span);
        }
      }
      throw std::runtime_error("in predicate: unknown symbol " + arg.getName() + "_");
    };
  }

  /**
   * This function is the same as createLambdaArgument(Symbol) except that it decomposes the column
   * and moves the matched span. This is used in 'Project'. Moving the columns is okay for the
   * following reasons: When using a storage engine the data in the Table (being projected from) is
   * already a shallow copy, so it is okay to move. In the case that the data is embedded in
   * the expression (e.g. in BOSSTests.cpp) then that table only exists within the expression
   * (within the Project operator) and therefore we must move from it. If we did a shallow copy, the
   * data would not exist outside of the project operator.
   */
  static Pred::Function createLambdaArgumentMove(Symbol const& arg) {
    return [arg](ExpressionArguments& columns,
                 Span<uint32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      for(auto& columnExpr : columns) {
        auto& column = get<ComplexExpression>(columnExpr);
        if(column.getHead().getName() == arg.getName()) {
          auto [unused1, unused2, unused3, spans] =
              std::move(get<ComplexExpression>(column.getArguments().at(0))).decompose();
          return std::visit(
              []<typename T>(Span<T>&& typedSpan) -> std::optional<ExpressionSpanArgument> {
                if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t> ||
                             std::is_same_v<T, std::string>) {
                  return Span<T>(std::move(typedSpan));
                } else {
                  throw std::runtime_error("unsupported column type in predicate");
                }
              },
              std::move(spans.at(0)));
        }
      }
      throw std::runtime_error("in predicate: unknown symbol " + arg.getName() + "_");
    };
  }

public:
  OperatorMap() {
    (*this)["Plus"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      ExpressionArguments args = input.getArguments();
      return visitAccumulate(std::move(args), static_cast<FirstArgument>(0),
                             [](auto&& state, auto&& arg) {
                               if constexpr(NumericType<std::decay_t<decltype(arg)>>) {
                                 state += arg;
                               }
                               return state;
                             });
    };
    (*this)["Plus"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::plus());
    };
    (*this)["Plus"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::plus());
    };

    (*this)["Times"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      ExpressionArguments args = input.getArguments();
      return visitAccumulate(std::move(args), static_cast<FirstArgument>(1),
                             [](auto&& state, auto&& arg) {
                               if constexpr(NumericType<std::decay_t<decltype(arg)>>) {
                                 state *= arg;
                               }
                               return state;
                             });
    };
    (*this)["Times"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::multiplies());
    };
    (*this)["Times"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::multiplies());
    };

    (*this)["Divide"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() == 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<Pred>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::divides());
      }
      return get<0>(input.getStaticArguments()) /
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Divide"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::divides());
    };
    (*this)["Divide"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::divides());
    };

    (*this)["Minus"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() == 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<Pred>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::minus());
      }
      return get<0>(input.getStaticArguments()) -
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Minus"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::minus());
    };
    (*this)["Minus"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::minus());
    };

    (*this)["Equal"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::equal_to());
    };
    (*this)["Equal"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::equal_to());
    };

    (*this)["Greater"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() == 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<Pred>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::greater());
      }
      return get<0>(input.getStaticArguments()) >
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Greater"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::greater());
    };
    (*this)["Greater"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::greater());
    };

    (*this)["And"_] = [](ComplexExpressionWithStaticArguments<Pred, Pred>&& input) -> Expression {
      return createLambdaPipelineOfExpressions(std::move(input));
    };

    (*this)["Where"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().empty());
      return Pred(get<0>(std::move(input).getStaticArguments()));
    };

    (*this)["As"_] = [](ComplexExpression&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() % 2 == 0);
      return std::move(input);
    };

    (*this)["DateObject"_] =
        [](ComplexExpressionWithStaticArguments<std::string>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().empty());
      auto str = get<0>(std::move(input).getStaticArguments());
      std::istringstream iss;
      iss.str(std::string(str));
      struct std::tm tm = {};
      iss >> std::get_time(&tm, "%Y-%m-%d");
      auto t = std::mktime(&tm);
      static int const hoursInADay = 24;
      // Q: How is a plain int like below stored as an Expression? Assume it is stored as an atomic
      // and can be accessed as an int normally would?
      return (int32_t)(std::chrono::duration_cast<std::chrono::hours>(
                           std::chrono::system_clock::from_time_t(t).time_since_epoch())
                           .count() /
                       hoursInADay);
    };

    (*this)["Project"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it));
      auto asExpr = std::move(*++it);
      if(relation.getHead().getName() != "Table") {
        return "Project"_(std::move(relation), std::move(asExpr));
      }
      auto columns = std::move(relation).getDynamicArguments();
      columns = transformDynamicsInColumnsToSpans(std::move(columns));
      ExpressionArguments asArgs = get<ComplexExpression>(std::move(asExpr)).getArguments();
      auto projectedColumns = ExpressionArguments(asArgs.size() / 2);
      size_t index = 0; // Process all calculation columns, each creating a new column
      for(auto asIt = std::make_move_iterator(asArgs.begin());
          asIt != std::make_move_iterator(asArgs.end()); ++asIt) {
        ++asIt;
        assert(std::holds_alternative<Symbol>(*asIt) || std::holds_alternative<Pred>(*asIt));
        if(std::holds_alternative<Pred>(*asIt)) {
          auto name = get<Symbol>(std::move(*--asIt));
          auto as = std::move(*++asIt);
          auto pred = get<Pred>(std::move(as));
          ExpressionSpanArguments spans{};
          if(auto projected = pred(columns, nullptr)) {
            std::visit([&spans](auto&& typedSpan) { spans.emplace_back(std::move(typedSpan)); },
                       std::move(*projected));
          }
          auto dynamics = ExpressionArguments{};
          dynamics.emplace_back(ComplexExpression("List"_, {}, {}, std::move(spans)));
          projectedColumns[index] = ComplexExpression(std::move(name), {}, std::move(dynamics), {});
        }
        ++index;
      }
      index = 0; // Process all symbol columns, each moving an existing column
      for(auto asIt = std::make_move_iterator(asArgs.begin());
          asIt != std::make_move_iterator(asArgs.end()); ++asIt) {
        if(std::holds_alternative<Symbol>(*++asIt)) {
          auto name = get<Symbol>(std::move(*--asIt));
          auto as = std::move(*++asIt);
          auto pred = createLambdaArgumentMove(get<Symbol>(std::move(as)));
          ExpressionSpanArguments spans{};
          if(auto projected = pred(columns, nullptr)) {
            std::visit([&spans](auto&& typedSpan) { spans.emplace_back(std::move(typedSpan)); },
                       std::move(*projected));
          }
          auto dynamics = ExpressionArguments{};
          dynamics.emplace_back(ComplexExpression("List"_, {}, {}, std::move(spans)));
          projectedColumns[index] = ComplexExpression(std::move(name), {}, std::move(dynamics), {});
        }
        ++index;
      }
      return ComplexExpression("Table"_, std::move(projectedColumns));
    };

    /** Currently candidate indexes is only used by Select which will create a new relation based
     * on the final set of indexes determined by the predicates. However, alternatively we could
     * simply pass the candidate indexes Span onto the next operator.
     */
    (*this)["Select"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it));
      auto predFunc = get<Pred>(std::move(*++it));
      auto columns = std::move(relation).getDynamicArguments();
      columns = transformDynamicsInColumnsToSpans(std::move(columns));
      if(auto predicate = predFunc(columns, nullptr)) {
        assert(std::holds_alternative<Span<uint32_t>>(*predicate));
        auto& indexes = std::get<Span<uint32_t>>(*predicate);
        const auto indexesPerThread = indexes.size() / DOP;
        // Threshold number of indexesPerThread at which point single threaded is faster
        if(DOP == 1 || indexesPerThread < 100) {
          for(auto& columnRef : columns) {
            auto column = get<ComplexExpression>(std::move(columnRef));
            auto [head, unused_, dynamics, spans] = std::move(column).decompose();
            auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
            auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
            listSpans.at(0) = std::visit(
                [&indexes]<typename T>(Span<T>&& typedSpan) -> ExpressionSpanArgument {
                  if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t> ||
                               std::is_same_v<T, std::string>) {
                    auto unfilteredColumn = std::move(typedSpan);
                    size_t size = indexes.size();
                    auto* filteredColumn = new T[size];
                    for(size_t i = 0; i < indexes.size(); ++i) {
                      filteredColumn[i] = unfilteredColumn[indexes[i]];
                    }
                    return Span<T>(filteredColumn, indexes.size(),
                                   [filteredColumn]() { delete[] filteredColumn; });
                  } else {
                    throw std::runtime_error("unsupported column type in select: " +
                                             std::string(typeid(T).name()));
                  }
                },
                std::move(listSpans.at(0)));
            dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                               std::move(listSpans));
            columnRef =
                ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
          }
        } else {
          /** Allocate memory for each new column sequentially, since jemalloc is not
           * expected to be beneficial for few large memory allocations in parallel
           */
          for(auto& columnRef : columns) {
            auto column = get<ComplexExpression>(std::move(columnRef));
            auto [head, unused_, dynamics, spans] = std::move(column).decompose();
            auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
            auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
            listSpans.at(0) = std::visit(
                [indexesPerThread,
                 &indexes]<typename T>(Span<T>&& typedSpan) -> ExpressionSpanArgument {
                  if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t> ||
                               std::is_same_v<T, std::string>) {
                    auto unfilteredColumn = std::move(typedSpan);
                    auto* filteredColumn = new T[indexes.size()];
                    uint32_t startIndex = 0;
                    for(uint32_t i = 0; i < DOP; ++i) {
                      uint32_t indexesToProcess =
                          i + 1 < DOP ? indexesPerThread : indexes.size() - startIndex;
                      ThreadPool::getInstance().enqueue(
                          [startIndex, endIndex = startIndex + indexesToProcess,
                           indexesPtr = &*indexes.begin(), filteredPtr = filteredColumn,
                           unfilteredPtr = &*unfilteredColumn.begin()]() {
                            for(uint32_t i = startIndex; i < endIndex; ++i) {
                              filteredPtr[i] = unfilteredPtr[indexesPtr[i]];
                            }
                          });
                      startIndex += indexesToProcess;
                    }
                    ThreadPool::getInstance().waitUntilComplete(DOP);
                    return Span<T>(filteredColumn, indexes.size(),
                                   [filteredColumn]() { delete[] filteredColumn; });
                  } else {
                    throw std::runtime_error("unsupported column type in select: " +
                                             std::string(typeid(T).name()));
                  }
                },
                std::move(listSpans.at(0)));
            dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                               std::move(listSpans));
            columnRef =
                ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
          }
        }
      }
      return ComplexExpression("Table"_, std::move(columns));
    };

    // TODO: Group currently only supports grouping on a single column (type long or double)
    (*this)["Group"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it++));
      auto columns = std::move(relation).getDynamicArguments();
      columns = transformDynamicsInColumnsToSpans(std::move(columns));

      auto resultColumns = ExpressionArguments{};

      auto byFlag = get<ComplexExpression>(args.at(1)).getHead().getName() == "By";
      std::map<long, std::vector<size_t>> map;
      if(byFlag) {
        auto byExpr = get<ComplexExpression>(std::move(*it++));
        auto groupingColumnSymbol = get<Symbol>(byExpr.getDynamicArguments().at(0));
        auto groupingPred = createLambdaArgument(groupingColumnSymbol);
        if(auto column = groupingPred(columns, nullptr)) {
          ExpressionSpanArgument span;
          std::visit(
              [&map, &span](auto&& typedSpan) {
                using Type = std::decay_t<decltype(typedSpan)>;
                if constexpr(std::is_same_v<Type, Span<int64_t>> ||
                             std::is_same_v<Type, Span<double>>) {
                  using ElementType = typename Type::element_type;
                  std::vector<ElementType> uniqueList;
                  for(size_t i = 0; i < typedSpan.size(); ++i) {
                    map[static_cast<long>(typedSpan[i])].push_back(i);
                  }
                  uniqueList.reserve(map.size());
                  for(const auto& pair : map) {
                    uniqueList.push_back(static_cast<ElementType>(pair.first));
                  }
                  span = Span<ElementType>(std::move(std::vector(uniqueList)));
                } else {
                  throw std::runtime_error("unsupported column type in group: " +
                                           std::string(typeid(typename Type::element_type).name()));
                }
              },
              std::move(*column));
          ExpressionSpanArguments spans{};
          spans.emplace_back(std::move(span));
          auto dynamics = ExpressionArguments{};
          dynamics.emplace_back(ComplexExpression("List"_, {}, {}, std::move(spans)));
          resultColumns.emplace_back(
              ComplexExpression(std::move(groupingColumnSymbol), {}, std::move(dynamics), {}));
        } else {
          throw std::runtime_error("couldn't access grouping column");
        }
      }

      auto asFlag = get<ComplexExpression>(args.at(1 + byFlag)).getHead().getName() == "As";
      if(asFlag) {
        auto asExpr = get<ComplexExpression>(std::move(*it));
        args = std::move(asExpr).getArguments();
        it = std::make_move_iterator(args.begin());
      }

      for(auto aggFuncIt = it; aggFuncIt != std::make_move_iterator(args.end()); ++aggFuncIt) {
        std::string specifiedName;
        if(asFlag) {
          auto name = get<Symbol>(std::move(*aggFuncIt++));
          specifiedName = name.getName();
        }
        auto aggFunc = get<ComplexExpression>(std::move(*aggFuncIt));
        assert(aggFunc.getDynamicArguments().size() == 1);
        auto columnSymbol = get<Symbol>(aggFunc.getDynamicArguments().at(0));
        auto aggFuncName = aggFunc.getHead().getName();
        auto pred = createLambdaArgument(columnSymbol);
        if(auto column = pred(columns, nullptr)) {
          ExpressionSpanArgument span;
          std::visit(
              [&span, &aggFuncName, &map, byFlag](auto&& typedSpan) {
                using Type = std::decay_t<decltype(typedSpan)>;
                if constexpr(std::is_same_v<Type, Span<int64_t>> ||
                             std::is_same_v<Type, Span<double>>) {
                  using ElementType = typename Type::element_type;
                  if(aggFuncName == "Sum") {
                    if(byFlag) {
                      std::vector<ElementType> results;
                      results.reserve(map.size());
                      for(auto const& pair : map) {
                        ElementType sum = 0;
                        for(auto const& index : pair.second) {
                          sum += typedSpan[index];
                        }
                        results.push_back(sum);
                      }
                      span = Span<ElementType>(std::move(std::vector(results)));
                    } else {
                      auto sum = std::accumulate(typedSpan.begin(), typedSpan.end(),
                                                 static_cast<ElementType>(0));
                      span = Span<ElementType>({sum});
                    }
                  } else if(aggFuncName == "Count") {
                    if(byFlag) {
                      std::vector<int64_t> results;
                      results.reserve(map.size());
                      for(auto const& pair : map) {
                        results.push_back(static_cast<int64_t>(pair.second.size()));
                      }
                      span = Span<int64_t>(std::vector(results));
                    } else {
                      auto count = static_cast<int64_t>(typedSpan.size());
                      span = Span<int64_t>({count});
                    }
                  } else {
                    throw std::runtime_error("unsupported aggregate function in group");
                  }
                } else {
                  throw std::runtime_error("unsupported column type in group: " +
                                           std::string(typeid(typename Type::element_type).name()));
                }
              },
              std::move(*column));
          ExpressionSpanArguments spans{};
          spans.emplace_back(std::move(span));
          auto dynamics = ExpressionArguments{};
          dynamics.emplace_back(ComplexExpression("List"_, {}, {}, std::move(spans)));
          auto name = asFlag ? Symbol(specifiedName) : columnSymbol;
          resultColumns.emplace_back(
              ComplexExpression(std::move(name), {}, std::move(dynamics), {}));
        } else {
          throw std::runtime_error("couldn't access aggregate column");
        }
      }
      return ComplexExpression("Table"_, std::move(resultColumns));
    };
  }
};

/***************************** BOSS API CONVENIENCE FUNCTIONS *****************************/
static Expression operator|(Expression&& expression, auto&& function) {
  return std::visit(boss::utilities::overload(std::move(function),
                                              [](auto&& atom) -> Expression { return atom; }),
                    std::move(expression));
}
/*****************************************************************************************/

static Expression evaluateInternal(Expression&& e) {
  static OperatorMap operators;
  return std::move(e) | [](ComplexExpression&& e) -> Expression {
    auto head = e.getHead();
    auto it = operators.find(head);
    if(it != operators.end())
      return it->second(std::move(e));
    return std::move(e);
  };
}

static boss::Expression evaluate(boss::Expression&& expr) {
  try {
    return toBOSSExpression(evaluateInternal(std::move(expr)));
  } catch(std::exception const& e) {
    boss::ExpressionArguments args;
    args.reserve(2);
    args.emplace_back(std::move(expr));
    args.emplace_back(std::string{e.what()});
    return boss::ComplexExpression{"ErrorWhenEvaluatingExpression"_, std::move(args)};
  }
}

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
}
