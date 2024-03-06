#include <Algorithm.hpp>
#include <ExpressionUtilities.hpp>

#include "HazardAdaptiveEngine.hpp"
#include "config.hpp"
#include "engineInstanceState.hpp"
#include "operators/partition.hpp"
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
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// #define DEBUG_MODE
// #define DEFER_TO_OTHER_ENGINE

using ExpressionBuilder = boss::utilities::ExtensibleExpressionBuilder<HAExpressionSystem>;
static ExpressionBuilder operator""_(const char* name, size_t /*unused*/) {
  return ExpressionBuilder(name);
}

using boss::expressions::CloneReason;
using BOSSExpressionSpanArguments = boss::DefaultExpressionSystem::ExpressionSpanArguments;
using BOSSExpressionSpanArgument = boss::DefaultExpressionSystem::ExpressionSpanArgument;

using adaptive::EngineInstanceState;
using adaptive::SelectOperatorState;
using adaptive::SelectOperatorStates;
using adaptive::config::nonVectorizedDOP;
using adaptive::config::partitionImplementation;
using adaptive::config::selectImplementation;

using boss::Span;
using boss::Symbol;
using SpanInputs = std::variant<std::vector<std::int32_t>, std::vector<std::int64_t>,
                                std::vector<std::double_t>, std::vector<std::string>>;

using namespace boss::algorithm;

static EngineInstanceState& getEngineInstanceState() {
  thread_local static EngineInstanceState engineInstanceState;
  return engineInstanceState;
}

/** Pred function takes a relation in the form of ExpressionArguments, and an optional pointer to
 * a span of candidate indexes. Based on these inputs and the internal predicate it returns an
 * optional span in the form of an ExpressionSpanArgument.
 */
class Pred : public std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&,
                                                                        Span<int32_t>*)> {
public:
  using Function =
      std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&, Span<int32_t>*)>;
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

// Use of wrapper class to ensure size of type added to variant is only 8 bytes
class PredWrapper {
public:
  explicit PredWrapper(std::unique_ptr<Pred> pred_) : pred(std::move(pred_)) {}

  PredWrapper(PredWrapper&& other) noexcept : pred(std::move(other.pred)) {}
  PredWrapper& operator=(PredWrapper&& other) noexcept {
    if(this != &other) {
      pred = std::move(other.pred);
    }
    return *this;
  }
  PredWrapper(PredWrapper const&) = delete;
  PredWrapper const& operator=(PredWrapper const&) = delete;
  ~PredWrapper() = default;

  Pred& getPred() {
    if(pred == nullptr) {
      std::cerr
          << "PredWrapper object does not own a Pred object and cannot be de-referenced. Exiting..."
          << std::endl;
      exit(1);
    }
    return *pred;
  }
  [[nodiscard]] const Pred& getPred() const {
    if(pred == nullptr) {
      std::cerr
          << "PredWrapper object does not own a Pred object and cannot be de-referenced. Exiting..."
          << std::endl;
      exit(1);
    }
    return *pred;
  }
  friend ::std::ostream& operator<<(std::ostream& out, PredWrapper const& predWrapper) {
    out << predWrapper.getPred();
    return out;
  }

private:
  std::unique_ptr<Pred> pred;
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

static boss::Expression toBOSSExpression(Expression&& expr, bool isPredicate = false) {
  return std::visit(
      boss::utilities::overload(
          [&](ComplexExpression&& e) -> boss::Expression {
            auto [head, unused_, dynamics, spans] = std::move(e).decompose();
            int NChildIsPredicate = dynamics.size(); // no child is a predicate as default
            if(head == "Select"_) {
              NChildIsPredicate = 1; // Select(relation, predicate)
            } else if(head == "Join"_) {
              NChildIsPredicate = 2; // Join(relation1, relation2, predicate)
            }
            boss::ExpressionArguments bossDynamics;
            bossDynamics.reserve(dynamics.size());
            std::transform(std::make_move_iterator(dynamics.begin()),
                           std::make_move_iterator(dynamics.end()),
                           std::back_inserter(bossDynamics), [&](auto&& arg) {
                             return toBOSSExpression(std::forward<decltype(arg)>(arg),
                                                     NChildIsPredicate-- == 0);
                           });
            BOSSExpressionSpanArguments bossSpans;
            bossSpans.reserve(spans.size());
            std::transform(
                std::make_move_iterator(spans.begin()), std::make_move_iterator(spans.end()),
                std::back_inserter(bossSpans), [](auto&& span) {
                  return std::visit(
                      []<typename T>(Span<T>&& typedSpan) -> BOSSExpressionSpanArgument {
                        if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                                     std::is_same_v<T, double_t> ||
                                     std::is_same_v<T, std::string> ||
                                     std::is_same_v<T, int32_t const> ||
                                     std::is_same_v<T, int64_t const> ||
                                     std::is_same_v<T, double_t const> ||
                                     std::is_same_v<T, std::string const>) {
                          return typedSpan;
                        } else {
                          throw std::runtime_error("span type not supported by BOSS core");
                        }
                      },
                      std::forward<decltype(span)>(span));
                });
            auto output = boss::ComplexExpression(std::move(head), {}, std::move(bossDynamics),
                                                  std::move(bossSpans));
            if(isPredicate && output.getHead() != "Where"_) {
              // make sure to re-inject "Where" clause before the predicate
              boss::ExpressionArguments whereArgs;
              whereArgs.emplace_back(std::move(output));
              return boss::ComplexExpression("Where"_, std::move(whereArgs));
            }
            return output;
          },
          [&](PredWrapper&& e) -> boss::Expression {
            // remaining unevaluated internal predicate, switch back to the initial expression
            auto output = static_cast<boss::Expression>(std::move(e.getPred()));
            if(isPredicate && (!std::holds_alternative<boss::ComplexExpression>(output) ||
                               std::get<boss::ComplexExpression>(output).getHead() != "Where"_)) {
              // make sure to re-inject "Where" clause before the predicate
              boss::ExpressionArguments whereArgs;
              whereArgs.emplace_back(std::move(output));
              return boss::ComplexExpression("Where"_, std::move(whereArgs));
            }
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
                      if constexpr(std::is_same_v<std::decay_t<decltype(a)>,
                                                  Span<PredWrapper const>>) {
                        throw std::runtime_error(
                            "Found a Span<PredWrapper const> in an expression to evaluate. "
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
          std::move(dispatchArgument));
    } else {
      ExpressionArguments rest{};
      if(dynamics1.size() > sizeof...(Args)) {
        std::transform(std::move_iterator(next(dynamics1.begin(), sizeof...(Args))),
                       std::move_iterator(dynamics1.end()), std::back_inserter(rest),
                       [](auto&& arg) { return std::forward<decltype(arg)>(arg); });
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
  requires !std::is_same_v<bool, std::remove_cv_t<T>>;
  requires std::is_arithmetic_v<decltype(param + 1)>;
  requires !std::is_pointer_v<T>;
};

template <typename T> concept IntegralType = requires(T param) {
  requires std::is_integral_v<T>;
  requires !std::is_same_v<bool, std::remove_cv_t<T>>;
  requires std::is_arithmetic_v<decltype(param + 1)>;
  requires !std::is_pointer_v<T>;
};

class OperatorMap : public std::unordered_map<boss::Symbol, StatelessOperator> {
private:
  template <typename T1, typename T2, typename F>
  static ExpressionSpanArgument createLambdaResult(T1&& arg1, T2&& arg2, F& f, int predicateID) {
    SelectOperatorState* state =
        predicateID >= 0 ? &getEngineInstanceState().getStateOfID(predicateID) : nullptr;
    auto engineDOP = getEngineInstanceState().getVectorizedDOP() == -1
                         ? nonVectorizedDOP
                         : getEngineInstanceState().getVectorizedDOP();
    ExpressionSpanArgument span;
    std::visit(boss::utilities::overload(
                   [](auto&& typedSpan1, auto&& typedSpan2) {
                     using SpanType1 = std::decay_t<decltype(typedSpan1)>;
                     using SpanType2 = std::decay_t<decltype(typedSpan2)>;
                     throw std::runtime_error(
                         "unsupported column types in lambda: " +
                         std::string(typeid(typename SpanType1::element_type).name()) + ", " +
                         std::string(typeid(typename SpanType2::element_type).name()));
                   },
                   [&span, &f, state, engineDOP]<NumericType Type1, NumericType Type2>(
                       boss::expressions::atoms::Span<Type1>&& typedSpan1,
                       boss::expressions::atoms::Span<Type2>&& typedSpan2) {
                     using OutputType = decltype(std::declval<decltype(f)>()(
                         std::declval<Type1>(), std::declval<Type2>()));
                     // If output of f is a bool, then we are performing a select so return indexes
                     if constexpr(std::is_same_v<OutputType, bool>) {
                       assert(typedSpan1.size() == 1 || typedSpan2.size() == 1);
                       if(typedSpan2.size() == 1) {
                         span = adaptive::select(selectImplementation, typedSpan1,
                                                 static_cast<Type1>(typedSpan2[0]), true, f, {},
                                                 engineDOP, state);
                       } else {
                         span = adaptive::select(selectImplementation, typedSpan2,
                                                 static_cast<Type1>(typedSpan1[0]), false, f, {},
                                                 engineDOP, state);
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
                   }),
               std::forward<T1>(arg1), std::forward<T2>(arg2));
    return span;
  }

  template <typename T1, typename T2, typename F>
  static ExpressionSpanArgument
  createLambdaPipelineResult(T1&& arg1, T2&& arg2, F& f, Span<int32_t>& indexes, int predicateID) {
    SelectOperatorState* state =
        predicateID >= 0 ? &getEngineInstanceState().getStateOfID(predicateID) : nullptr;
    auto engineDOP = getEngineInstanceState().getVectorizedDOP() == -1
                         ? nonVectorizedDOP
                         : getEngineInstanceState().getVectorizedDOP();
    std::visit(boss::utilities::overload(
                   [](auto&& typedSpan1, auto&& typedSpan2) {
                     using SpanType1 = std::decay_t<decltype(typedSpan1)>;
                     using SpanType2 = std::decay_t<decltype(typedSpan2)>;
                     throw std::runtime_error(
                         "unsupported column types in lambda: " +
                         std::string(typeid(typename SpanType1::element_type).name()) + ", " +
                         std::string(typeid(typename SpanType2::element_type).name()));
                   },
                   [&indexes, &f, state, engineDOP]<NumericType Type1, NumericType Type2>(
                       boss::expressions::atoms::Span<Type1>&& typedSpan1,
                       boss::expressions::atoms::Span<Type2>&& typedSpan2) {
                     using OutputType = decltype(std::declval<decltype(f)>()(
                         std::declval<Type1>(), std::declval<Type2>()));

                     if constexpr(std::is_same_v<OutputType, bool>) {
                       assert(typedSpan1.size() == 1 || typedSpan2.size() == 1);
                       if(typedSpan2.size() == 1) {
                         indexes = adaptive::select(selectImplementation, typedSpan1,
                                                    static_cast<Type1>(typedSpan2[0]), true, f,
                                                    std::move(indexes), engineDOP, state);
                       } else {
                         indexes = adaptive::select(selectImplementation, typedSpan2,
                                                    static_cast<Type2>(typedSpan1[0]), false, f,
                                                    std::move(indexes), engineDOP, state);
                       }
                     } else {
                       throw std::runtime_error(
                           "function in createLambdaPipelineResult does not return bool: " +
                           std::string(typeid(Type1).name()));
                     }
                   }),
               std::forward<T1>(arg1), std::forward<T2>(arg2));
    return {};
  }

  template <typename T, typename F>
  static PredWrapper createLambdaExpression(ComplexExpressionWithStaticArguments<T>&& e, F&& f) {
    assert(e.getSpanArguments().empty());
    assert(e.getDynamicArguments().size() >= 1);
    auto numDynamicArgs = e.getDynamicArguments().size();
    int predicateID = -1;
    if(numDynamicArgs > 1 &&
       std::holds_alternative<ComplexExpression>(e.getDynamicArguments().at(1)) &&
       get<ComplexExpression>(e.getDynamicArguments().at(1)).getHead() == "PredicateID"_) {
      numDynamicArgs = 1;
      predicateID = get<int>(
          get<ComplexExpression>(e.getDynamicArguments().at(1)).getDynamicArguments().at(0));
    }
    if(numDynamicArgs == 1) {
      Pred::Function pred = std::visit(
          [&e, &f, predicateID](auto&& arg) -> Pred::Function {
            return [pred1 = createLambdaArgument(get<0>(e.getStaticArguments())),
                    pred2 = createLambdaArgument(arg), &f, predicateID](
                       ExpressionArguments& columns,
                       Span<int32_t>* indexes) mutable -> std::optional<ExpressionSpanArgument> {
              auto arg1 = pred1(columns, nullptr);
              auto arg2 = pred2(columns, nullptr);
              if(!arg1 || !arg2) {
                return std::nullopt;
              }
              if(indexes) {
                return createLambdaPipelineResult(std::move(*arg1), std::move(*arg2), f, *indexes,
                                                  predicateID);
              }
              return createLambdaResult(std::move(*arg1), std::move(*arg2), f, predicateID);
            };
          },
          e.getDynamicArguments().at(0));
      return PredWrapper(std::make_unique<Pred>(std::move(pred), toBOSSExpression(std::move(e))));
    } else {
      Pred::Function pred = std::accumulate(
          e.getDynamicArguments().begin(), e.getDynamicArguments().end(),
          (Pred::Function&&)createLambdaArgument(get<0>(e.getStaticArguments())),
          [&f](auto&& acc, auto const& e) -> Pred::Function {
            return std::visit(
                [&f, &acc](auto&& arg) -> Pred::Function {
                  return
                      [acc, pred2 = createLambdaArgument(arg), &f](
                          ExpressionArguments& columns,
                          Span<int32_t>* indexes) mutable -> std::optional<ExpressionSpanArgument> {
                        assert(!indexes);
                        auto arg1 = acc(columns, nullptr);
                        auto arg2 = pred2(columns, nullptr);
                        if(!arg1 || !arg2) {
                          return {};
                        }
                        return createLambdaResult(std::move(*arg1), std::move(*arg2), f, -1);
                      };
                },
                e);
          });
      return PredWrapper(std::make_unique<Pred>(std::move(pred), toBOSSExpression(std::move(e))));
    }
  }

  template <typename T1, typename T2>
  static PredWrapper
  createLambdaPipelineOfExpressions(ComplexExpressionWithStaticArguments<T1, T2>&& e) {
    assert(e.getSpanArguments().empty());
    std::vector<Pred::Function> preds;
    preds.reserve(2 + e.getDynamicArguments().size());
    preds.push_back(createLambdaArgument(get<0>(e.getStaticArguments())));
    preds.push_back(createLambdaArgument(get<1>(e.getStaticArguments())));
    for(auto it = e.getDynamicArguments().begin(); it != e.getDynamicArguments().end(); ++it) {
      preds.push_back(std::move(get<PredWrapper>(*it).getPred()));
    }
    Pred::Function pred =
        [preds = std::move(preds)](
            ExpressionArguments& columns,
            Span<int32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      auto candidateIndexes = preds[0](columns, nullptr);
      if(!candidateIndexes.has_value()) {
        return std::nullopt;
      }
      ExpressionSpanArgument span;
      std::visit(
          [&span, &preds, &columns](auto&& candidateIndexes) {
            if constexpr(std::is_same_v<std::decay_t<decltype(candidateIndexes)>, Span<int32_t>>) {
              for(auto it = std::next(preds.begin()); it != preds.end(); ++it) {
                if(candidateIndexes.size() == 0) {
                  break;
                }
                (*it)(columns, &candidateIndexes);
              }
              span = Span<int32_t>(std::move(candidateIndexes));
            } else {
              throw std::runtime_error("Multi-predicate lambda column input is not indexes");
            }
          },
          std::move(*candidateIndexes));
      return span;
    };
    return PredWrapper(std::make_unique<Pred>(std::move(pred), toBOSSExpression(std::move(e))));
  }

  /** Turns single arguments into a single element span so that createLambdaResult acts on two
   *  spans. Could make this cleaner by Pred::Function returning an Expression or variant (but would
   *  require more code)?
   */
  template <typename ArgType> static Pred::Function createLambdaArgument(ArgType const& arg) {
    if constexpr(NumericType<ArgType>) {
      return [arg](ExpressionArguments& /*unused*/,
                   Span<int32_t>* /*unused*/) -> std::optional<ExpressionSpanArgument> {
        return Span<ArgType>(std::move(std::vector({arg})));
      };
    } else {
      throw std::runtime_error("unsupported argument type in predicate");
      return [](ExpressionArguments& /*unused*/,
                Span<int32_t>* /*unused*/) -> std::optional<ExpressionSpanArgument> { return {}; };
    }
  }

  static Pred::Function createLambdaArgument(PredWrapper const& arg) {
    return [f = static_cast<Pred::Function const&>(arg.getPred())](ExpressionArguments& columns,
                                                                   Span<int32_t>* indexes) {
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
    return [arg, index = 0U](
               ExpressionArguments& columns,
               Span<int32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      for(auto& columnExpr : columns) {
        auto& column = get<ComplexExpression>(columnExpr);
        if(column.getHead().getName() == arg.getName()) {
          if(index >=
             get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments().size()) {
            return std::nullopt;
          }
          auto& span =
              get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments().at(index++);
          return std::visit(
              []<typename T>(Span<T> const& typedSpan) -> std::optional<ExpressionSpanArgument> {
                if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                             std::is_same_v<T, double_t> || std::is_same_v<T, std::string> ||
                             std::is_same_v<T, int32_t const> || std::is_same_v<T, int64_t const> ||
                             std::is_same_v<T, double_t const> ||
                             std::is_same_v<T, std::string const>) {
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
    return [arg, index = 0U](
               ExpressionArguments& columns,
               Span<int32_t>* /*unused*/) mutable -> std::optional<ExpressionSpanArgument> {
      for(auto& columnExpr : columns) {
        auto& column = get<ComplexExpression>(columnExpr);
        if(column.getHead().getName() == arg.getName()) {
          if(index >=
             get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments().size()) {
            return std::nullopt;
          }
          auto [head, unused_, dynamics, spans] = std::move(column).decompose();
          auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
          auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
          auto span = std::move(listSpans.at(index++));
          dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                             std::move(listSpans));
          columnExpr =
              ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
          return span;
        }
      }
      throw std::runtime_error("in predicate: unknown symbol " + arg.getName() + "_");
    };
  }

  static ComplexExpression constructTableAndRemoveColumn(ExpressionArguments&& columns,
                                                         const Symbol& columnToRemove) {
    ExpressionArguments args;
    for(auto&& column : columns) {
      if(get<ComplexExpression>(column).getHead().getName() != columnToRemove.getName()) {
        args.emplace_back(std::move(column));
      }
    }
    return {"Table"_, {}, std::move(args), {}};
  }

  static ComplexExpression constructTableWithEmptyColumns(ExpressionArguments&& columns) {
    ExpressionArguments args;
    for(auto&& column : columns) {
      ExpressionArguments dyns;
      dyns.emplace_back("List"_());
      args.emplace_back(ComplexExpression(std::move(get<ComplexExpression>(column)).getHead(), {},
                                          std::move(dyns), {}));
    }
    return {"Table"_, {}, std::move(args), {}};
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
    (*this)["Plus"_] = [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
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
    (*this)["Times"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::multiplies());
    };

    (*this)["Divide"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() == 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<PredWrapper>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::divides());
      }
      return get<0>(input.getStaticArguments()) /
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Divide"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::divides());
    };
    (*this)["Divide"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::divides());
    };

    (*this)["Minus"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() == 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<PredWrapper>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::minus());
      }
      return get<0>(input.getStaticArguments()) -
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Minus"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::minus());
    };
    (*this)["Minus"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::minus());
    };

    (*this)["Equal"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::equal_to());
    };
    (*this)["Equal"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::equal_to());
    };

    (*this)["Greater"_] =
        []<NumericType FirstArgument>(
            ComplexExpressionWithStaticArguments<FirstArgument>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().size() >= 1);
      if(std::holds_alternative<Symbol>(input.getDynamicArguments().at(0)) ||
         std::holds_alternative<PredWrapper>(input.getDynamicArguments().at(0))) {
        return createLambdaExpression(std::move(input), std::greater());
      }
      return get<0>(input.getStaticArguments()) >
             get<FirstArgument>(input.getDynamicArguments().at(0));
    };
    (*this)["Greater"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::greater());
    };
    (*this)["Greater"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::greater());
    };

    (*this)["And"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper, PredWrapper>&& input) -> Expression {
      return createLambdaPipelineOfExpressions(std::move(input));
    };

    (*this)["Where"_] =
        [](ComplexExpressionWithStaticArguments<PredWrapper>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().empty());
      return PredWrapper(get<0>(std::move(input).getStaticArguments()));
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
      return (int32_t)(std::chrono::duration_cast<std::chrono::hours>(
                           std::chrono::system_clock::from_time_t(t).time_since_epoch())
                           .count() /
                       hoursInADay);
    };

    (*this)["Project"_] = [](ComplexExpression&& inputExpr) -> Expression {
      if(!holds_alternative<ComplexExpression>(*(inputExpr.getArguments().begin())) ||
         get<ComplexExpression>(*(inputExpr.getArguments().begin())).getHead().getName() !=
             "Table") {
#ifdef DEBUG_MODE
        std::cout << "Table is invalid, Project left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it));
      auto asExpr = std::move(*++it);
      auto columns = std::move(relation).getDynamicArguments();
      ExpressionArguments asArgs = get<ComplexExpression>(std::move(asExpr)).getArguments();
      auto projectedColumns = ExpressionArguments(asArgs.size() / 2);
      size_t index = 0; // Process all calculation columns, each creating a new column
      for(auto asIt = std::make_move_iterator(asArgs.begin());
          asIt != std::make_move_iterator(asArgs.end()); ++asIt) {
        ++asIt;
        assert(std::holds_alternative<Symbol>(*asIt) || std::holds_alternative<PredWrapper>(*asIt));
        if(std::holds_alternative<PredWrapper>(*asIt)) {
          auto name = get<Symbol>(std::move(*--asIt));
          auto as = std::move(*++asIt);
          auto predWrapper = get<PredWrapper>(std::move(as));
          auto& pred = predWrapper.getPred();
          ExpressionSpanArguments spans{};
          while(auto projected = pred(columns, nullptr)) {
            spans.push_back(std::move(*projected));
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
          while(auto projected = pred(columns, nullptr)) {
            spans.push_back(std::move(*projected));
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
      if(!holds_alternative<ComplexExpression>(*(inputExpr.getArguments().begin())) ||
         get<ComplexExpression>(*(inputExpr.getArguments().begin())).getHead().getName() !=
             "Table") {
#ifdef DEBUG_MODE
        std::cout << "Table is invalid, Select left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      if(!holds_alternative<PredWrapper>(*++(inputExpr.getArguments().begin()))) {
#ifdef DEBUG_MODE
        std::cout << "Predicate is invalid, Select left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it));
      auto predWrapper = get<PredWrapper>(std::move(*++it));
      auto& predFunc = predWrapper.getPred();
      auto columns = std::move(relation).getDynamicArguments();
#ifdef DEFER_TO_OTHER_ENGINE
      ExpressionSpanArguments indexesArg;
      while(auto predicate = predFunc(columns, nullptr)) {
        assert(std::holds_alternative<Span<int32_t>>(*predicate));
        auto& indexes = std::get<Span<int32_t>>(*predicate);
        indexesArg.emplace_back(std::move(indexes));
      }
      auto tableExpression = ComplexExpression("Table"_, std::move(columns));
      ExpressionArguments tableArg;
      tableArg.emplace_back(std::move(tableExpression));
      return ComplexExpression("Gather"_, {}, std::move(tableArg), std::move(indexesArg));
#else
      ExpressionSpanArguments indexesArg;
      while(auto predicate = predFunc(columns, nullptr)) {
        assert(std::holds_alternative<Span<int32_t>>(*predicate));
        auto& indexes = std::get<Span<int32_t>>(*predicate);
        indexesArg.emplace_back(std::move(indexes));
      }
      for(auto& columnRef : columns) {
        auto column = get<ComplexExpression>(std::move(columnRef));
        auto [head, unused_, dynamics, spans] = std::move(column).decompose();
        auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
        auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
        for(size_t spanNum = 0; spanNum < listSpans.size(); ++spanNum) {
          const auto& indexes = std::get<Span<int32_t>>(indexesArg.at(spanNum));
          listSpans.at(spanNum) = std::visit(
              [&indexes]<typename T>(Span<T>&& typedSpan) -> ExpressionSpanArgument {
                if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                             std::is_same_v<T, double_t> || std::is_same_v<T, std::string> ||
                             std::is_same_v<T, int32_t const> || std::is_same_v<T, int64_t const> ||
                             std::is_same_v<T, double_t const> ||
                             std::is_same_v<T, std::string const>) {
                  auto unfilteredColumn = std::move(typedSpan);
                  auto* filteredColumn = new std::remove_cv_t<T>[indexes.size()];
                  if(nonVectorizedDOP == 1 ||
                     indexes.size() <= (2 * adaptive::config::minTuplesPerThread)) {
                    for(size_t i = 0; i < indexes.size(); ++i) {
                      filteredColumn[i] = unfilteredColumn[indexes[i]];
                    }
                  } else {
                    const auto numThreads =
                        std::min(nonVectorizedDOP,
                                 static_cast<int32_t>(indexes.size() /
                                                      adaptive::config::minTuplesPerThread));
                    assert(numThreads >= 2);
                    const auto indexesPerThread = indexes.size() / numThreads;
                    int32_t startIndex = 0;
                    for(int32_t threadNum = 0; threadNum < numThreads; ++threadNum) {
                      int32_t indexesToProcess =
                          threadNum + 1 < numThreads
                              ? static_cast<int32_t>(indexesPerThread)
                              : static_cast<int32_t>(indexes.size()) - startIndex;
                      ThreadPool::getInstance().enqueue(
                          [startIndex, endIndex = startIndex + indexesToProcess,
                           indexesPtr = &*indexes.begin(), filteredPtr = filteredColumn,
                           unfilteredPtr = &*unfilteredColumn.begin()]() {
                            for(int32_t i = startIndex; i < endIndex; ++i) {
                              filteredPtr[i] = unfilteredPtr[indexesPtr[i]];
                            }
                          });
                      startIndex += indexesToProcess;
                    }
                    ThreadPool::getInstance().waitUntilComplete(numThreads);
                  }
                  return Span<T>(filteredColumn, indexes.size(),
                                 [filteredColumn]() { delete[] filteredColumn; });
                } else {
                  throw std::runtime_error("unsupported column type in select: " +
                                           std::string(typeid(T).name()));
                }
              },
              std::move(listSpans.at(spanNum)));
        }
        dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                           std::move(listSpans));
        columnRef = ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
      }
      return ComplexExpression("Table"_, std::move(columns));
#endif
    };

    /** Currently only partitions each table on a single 'key' column
     */
    (*this)["Join"_] = [](ComplexExpression&& inputExpr) -> Expression {
      if(!holds_alternative<ComplexExpression>(*(inputExpr.getArguments().begin())) ||
         get<ComplexExpression>(*(inputExpr.getArguments().begin())).getHead().getName() !=
             "Table") {
#ifdef DEBUG_MODE
        std::cout << "Left join Table is invalid, Join left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      if(!holds_alternative<ComplexExpression>(*++(inputExpr.getArguments().begin())) ||
         get<ComplexExpression>(*++(inputExpr.getArguments().begin())).getHead().getName() !=
             "Table") {
#ifdef DEBUG_MODE
        std::cout << "Right join Table is invalid, Join left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      if(!holds_alternative<PredWrapper>(*std::next(inputExpr.getArguments().begin(), 2))) {
#ifdef DEBUG_MODE
        std::cout << "Predicate is invalid, Join left unevaluated..." << std::endl;
#endif
        return toBOSSExpression(std::move(inputExpr));
      }
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto leftRelation = get<ComplexExpression>(std::move(*it));
      auto leftRelationColumns = std::move(leftRelation).getDynamicArguments();
      auto rightRelation = get<ComplexExpression>(std::move(*++it));
      auto rightRelationColumns = std::move(rightRelation).getDynamicArguments();
      auto predWrapper = get<PredWrapper>(std::move(*++it));
      auto predTmp = static_cast<boss::Expression>(std::move(predWrapper.getPred()));
      Expression pred = std::move(predTmp);
      auto& predExpr = get<ComplexExpression>(pred);
      auto& leftKeySymbol = get<Symbol>(predExpr.getDynamicArguments().at(0));
      auto& rightKeySymbol = get<Symbol>(predExpr.getDynamicArguments().at(1));

      if(leftRelationColumns.empty()) {
        return constructTableWithEmptyColumns(std::move(rightRelationColumns));
      }
      if(rightRelationColumns.empty()) {
        return constructTableWithEmptyColumns(std::move(leftRelationColumns));
      }

      auto getColumnSpans = [](ExpressionArguments& columns,
                               const Symbol& columnSymbol) -> const ExpressionSpanArguments& {
        for(auto& columnExpr : columns) {
          auto& column = get<ComplexExpression>(columnExpr);
          if(column.getHead().getName() == columnSymbol.getName()) {
            return get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments();
          }
        }
        throw std::runtime_error("column name not in relation: " + columnSymbol.getName());
      };

      auto& leftKeySpans = getColumnSpans(leftRelationColumns, leftKeySymbol);
      auto& rightKeySpans = getColumnSpans(rightRelationColumns, rightKeySymbol);

      if(leftKeySpans.empty() || rightKeySpans.empty()) {
        leftRelationColumns.insert(leftRelationColumns.end(),
                                   std::make_move_iterator(rightRelationColumns.begin()),
                                   std::make_move_iterator(rightRelationColumns.end()));
        return constructTableWithEmptyColumns(std::move(leftRelationColumns));
      }

      auto engineDOP = getEngineInstanceState().getVectorizedDOP() == -1
                           ? nonVectorizedDOP
                           : getEngineInstanceState().getVectorizedDOP();

      auto partitionedTables = std::visit(
          boss::utilities::overload(
              [](auto&& typedSpan1, auto&& typedSpan2) -> adaptive::PartitionedJoinArguments {
                using SpanType1 = std::decay_t<decltype(typedSpan1)>;
                using SpanType2 = std::decay_t<decltype(typedSpan2)>;
                throw std::runtime_error(
                    "Join key has at least one unsupported column type: " +
                    std::string(typeid(typename SpanType1::element_type).name()) + ", " +
                    std::string(typeid(typename SpanType2::element_type).name()));
              },
              [&leftKeySpans, &rightKeySpans, engineDOP]<IntegralType Type1, IntegralType Type2>(
                  boss::expressions::atoms::Span<Type1> const& /*typedSpan1*/,
                  boss::expressions::atoms::Span<Type2> const& /*typedSpan2*/) {
                return adaptive::partitionJoinExpr<Type2, Type2>(
                    partitionImplementation, leftKeySpans, rightKeySpans, engineDOP);
              }),
          leftKeySpans.at(0), rightKeySpans.at(0));

      auto& tableOnePartitionsOfKeySpans = partitionedTables.tableOnePartitionsOfKeySpans;
      auto& tableOnePartitionsOfIndexSpans = partitionedTables.tableOnePartitionsOfIndexSpans;
      auto& tableTwoPartitionsOfKeySpans = partitionedTables.tableTwoPartitionsOfKeySpans;
      auto& tableTwoPartitionsOfIndexSpans = partitionedTables.tableTwoPartitionsOfIndexSpans;
#ifdef DEBUG_MODE
      assert(tableOnePartitionsOfKeySpans.size() == tableOnePartitionsOfIndexSpans.size() &&
             tableOnePartitionsOfIndexSpans.size() == tableTwoPartitionsOfKeySpans.size() &&
             tableTwoPartitionsOfKeySpans.size() == tableTwoPartitionsOfIndexSpans.size());
#endif

      ExpressionArguments leftArgs, rightArgs, joinArgs;
      leftArgs.emplace_back(
          constructTableAndRemoveColumn(std::move(leftRelationColumns), leftKeySymbol));
      rightArgs.emplace_back(
          constructTableAndRemoveColumn(std::move(rightRelationColumns), rightKeySymbol));

      for(size_t i = 0; i < tableOnePartitionsOfKeySpans.size(); ++i) {
        ExpressionArguments leftKeyList, leftPartitionArg, rightKeyList, rightPartitionArg;
#ifdef DEBUG_MODE
        std::cout << "Number of tableOneKeySpans in partition " << i << " = "
                  << tableOnePartitionsOfKeySpans.at(i).size() << std::endl;
        std::cout << "Number of tableOneIndexesSpans in partition " << i << " = "
                  << tableOnePartitionsOfIndexSpans.at(i).size() << std::endl;
        std::cout << "Number of tableTwoKeySpans in partition " << i << " = "
                  << tableTwoPartitionsOfKeySpans.at(i).size() << std::endl;
        std::cout << "Number of tableTwoIndexesSpans in partition " << i << " = "
                  << tableTwoPartitionsOfIndexSpans.at(i).size() << std::endl;
#endif
        leftKeyList.emplace_back(
            ComplexExpression("List"_, {}, {}, std::move(tableOnePartitionsOfKeySpans.at(i))));
        leftPartitionArg.emplace_back(
            ComplexExpression(leftKeySymbol, {}, std::move(leftKeyList), {}));
        leftPartitionArg.emplace_back(
            ComplexExpression("Indexes"_, {}, {}, std::move(tableOnePartitionsOfIndexSpans.at(i))));
        leftArgs.emplace_back(ComplexExpression("Partition"_, {}, std::move(leftPartitionArg), {}));

        rightKeyList.emplace_back(
            ComplexExpression("List"_, {}, {}, std::move(tableTwoPartitionsOfKeySpans.at(i))));
        rightPartitionArg.emplace_back(
            ComplexExpression(rightKeySymbol, {}, std::move(rightKeyList), {}));
        rightPartitionArg.emplace_back(
            ComplexExpression("Indexes"_, {}, {}, std::move(tableTwoPartitionsOfIndexSpans.at(i))));
        rightArgs.emplace_back(
            ComplexExpression("Partition"_, {}, std::move(rightPartitionArg), {}));
      }

      auto leftExpr = ComplexExpression("RadixPartition"_, {}, std::move(leftArgs), {});
      auto rightExpr = ComplexExpression("RadixPartition"_, {}, std::move(rightArgs), {});

      joinArgs.emplace_back(std::move(leftExpr));
      joinArgs.emplace_back(std::move(rightExpr));
      joinArgs.emplace_back(std::move(predExpr));

      return ComplexExpression("Join"_, {}, std::move(joinArgs), {});
    };

#ifndef DEFER_TO_OTHER_ENGINE
    // TODO: Group currently does not handle multiple spans
    // TODO: Group currently only supports grouping on a single column (type long or double)
    (*this)["Group"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = get<ComplexExpression>(std::move(*it++));
      auto columns = std::move(relation).getDynamicArguments();
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
              [&](auto&& typedSpan) {
                using Type = std::decay_t<decltype(typedSpan)>;
                if constexpr(std::is_same_v<Type, Span<int32_t>> ||
                             std::is_same_v<Type, Span<int64_t>> ||
                             std::is_same_v<Type, Span<double>> ||
                             std::is_same_v<Type, Span<int32_t const>> ||
                             std::is_same_v<Type, Span<int64_t const>> ||
                             std::is_same_v<Type, Span<double const>>) {
                  using ElementType = typename Type::element_type;
                  std::vector<std::remove_cv_t<ElementType>> uniqueList;
                  while(true) {
                    for(size_t i = 0; i < typedSpan.size(); ++i) {
                      map[static_cast<long>(typedSpan[i])].push_back(i);
                    }
                    auto column = groupingPred(columns, nullptr);
                    if(!column.has_value()) {
                      break;
                    }
                    typedSpan = std::get<Span<ElementType>>(std::move(*column));
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
              [&](auto&& typedSpan) {
                using Type = std::decay_t<decltype(typedSpan)>;
                if constexpr(std::is_same_v<Type, Span<int32_t>> ||
                             std::is_same_v<Type, Span<int64_t>> ||
                             std::is_same_v<Type, Span<double>> ||
                             std::is_same_v<Type, Span<int32_t const>> ||
                             std::is_same_v<Type, Span<int64_t const>> ||
                             std::is_same_v<Type, Span<double const>>) {
                  using ElementType = typename Type::element_type;
                  if(aggFuncName == "Sum") {
                    // TODO: Group "Sum By" currently only works with single span columns
                    if(byFlag) {
                      std::vector<std::remove_cv_t<ElementType>> results;
                      results.reserve(map.size());
                      for(auto const& pair : map) {
                        std::remove_cv_t<ElementType> sum = 0;
                        for(auto const& index : pair.second) {
                          sum += typedSpan[index];
                        }
                        results.push_back(sum);
                      }
                      span = Span<ElementType>(std::move(std::vector(results)));
                    } else {
                      std::remove_cv_t<ElementType> sum = 0;
                      while(true) {
                        sum += std::accumulate(typedSpan.begin(), typedSpan.end(),
                                               static_cast<ElementType>(0));
                        auto column = pred(columns, nullptr);
                        if(!column.has_value()) {
                          break;
                        }
                        typedSpan = std::get<Span<ElementType>>(std::move(*column));
                      }
                      span = Span<ElementType>({sum});
                    }
                  } else if(aggFuncName == "Count") {
                    if(byFlag) {
                      std::vector<int32_t> results;
                      results.reserve(map.size());
                      for(auto const& pair : map) {
                        results.push_back(static_cast<int32_t>(pair.second.size()));
                      }
                      span = Span<int32_t>(std::vector(results));
                    } else {
                      int32_t count = 0;
                      while(true) {
                        count += static_cast<int32_t>(typedSpan.size());
                        auto column = pred(columns, nullptr);
                        if(!column.has_value()) {
                          break;
                        }
                        typedSpan = std::get<Span<ElementType>>(std::move(*column));
                      }
                      span = Span<int32_t>({count});
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
#endif
  }
};

static Expression evaluateInternal(Expression&& e) {
  static OperatorMap operators;
  if(std::holds_alternative<ComplexExpression>(e)) {
    auto [head, unused_, dynamics, spans] = get<ComplexExpression>(std::move(e)).decompose();
    ExpressionArguments evaluatedDynamics;
    evaluatedDynamics.reserve(dynamics.size());
    std::transform(std::make_move_iterator(dynamics.begin()),
                   std::make_move_iterator(dynamics.end()), std::back_inserter(evaluatedDynamics),
                   [](auto&& arg) { return evaluateInternal(std::forward<decltype(arg)>(arg)); });
    auto unevaluated =
        ComplexExpression(std::move(head), {}, std::move(evaluatedDynamics), std::move(spans));
    auto it = operators.find(unevaluated.getHead());
    if(it != operators.end())
      return it->second(std::move(unevaluated));
    return std::move(unevaluated);
  }
  return std::move(e);
}

static boss::Expression evaluate(boss::Expression&& expr) {
  try {
#ifdef DEBUG_MODE
    std::cout << "Input expression: "
              << utilities::injectDebugInfoToSpans(expr.clone(CloneReason::FOR_TESTING))
              << std::endl;
#endif
    while(std::holds_alternative<boss::ComplexExpression>(expr) &&
          get<boss::ComplexExpression>(expr).getHead() == "Let"_) {
      auto [head, unused_, dynamics, spans] =
          std::move(get<boss::ComplexExpression>(expr)).decompose();
      expr = std::move(dynamics.at(0));
      auto letExpr = get<boss::ComplexExpression>(std::move(dynamics.at(1)));
      if(letExpr.getHead().getName() == "Stats") {
        auto [statsHead, unused1, statsArgs, unused2] = std::move(letExpr).decompose();
        if(statsArgs.empty())
          throw std::runtime_error("No pointers to operator states in stats expression");
        auto selectOperatorStates =
            reinterpret_cast<SelectOperatorStates*>(get<int64_t>(statsArgs.at(0)));
        getEngineInstanceState().setStatsPtr(selectOperatorStates);
      } else if(letExpr.getHead().getName() == "Parallel") {
        auto [parallelHead, unused1, parallelArgs, unused2] = std::move(letExpr).decompose();
        if(parallelArgs.empty())
          throw std::runtime_error("No constantsDOP value in parallel expression");
        getEngineInstanceState().setVectorizedDOP(get<int32_t>(parallelArgs.at(0)));
      } else {
        throw std::runtime_error("Unexpected argument of 'Let' expression: " +
                                 letExpr.getHead().getName());
      }
    }
    return toBOSSExpression(evaluateInternal(std::move(expr)));
  } catch(std::exception const& e) {
    boss::ExpressionArguments args;
    args.reserve(2);
    args.emplace_back(std::move(expr));
    args.emplace_back(std::string{e.what()});
    return boss::ComplexExpression{"ErrorWhenEvaluatingExpression"_, std::move(args)};
  }
}

extern "C" __attribute__((visibility("default"))) BOSSExpression* evaluate(BOSSExpression* e) {
  return new BOSSExpression{.delegate = evaluate(std::move(e->delegate))};
}
