#include <Algorithm.hpp>
#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>

#include <algorithm>
#include <any>
#include <cassert>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

class Pred;

using HAExpressionSystem = boss::expressions::generic::ExtensibleExpressionSystem<Pred>;
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
};

using boss::Span;
using boss::Symbol;
using SpanInputs =
    std::variant<std::vector<std::int64_t>, std::vector<std::double_t>, std::vector<std::string>>;

using namespace boss::algorithm;

class Pred : public std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&)> {
public:
  using Function = std::function<std::optional<ExpressionSpanArgument>(ExpressionArguments&)>;
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

// Q: Should tests all use spans for data - currently most using dynamics which requires this func
template <typename... StaticArgumentTypes>
ComplexExpressionWithStaticArguments<StaticArgumentTypes...>
transformDynamicsToSpans(ComplexExpressionWithStaticArguments<StaticArgumentTypes...>&& input_) {
  std::vector<SpanInputs> spanInputs;
  auto [head, statics, dynamics, oldSpans] = std::move(input_).decompose();

  //  std::cout << "TransformDynamicsToSpans head: " << head << std::endl;
  //  std::cout << "Dynamics to transform: " << dynamics.size() << std::endl;
  //  if (oldSpans.begin() != oldSpans.end()) {
  //    std::cout << "Current spans: " << std::endl;
  //    std::visit(
  //        [](auto&& arg) {
  //          using T = std::decay_t<decltype(arg)>;
  //          if constexpr(std::is_same_v<T, boss::Span<long>>) {
  //            for(const auto& value : arg) {
  //              std::cout << value << " ";
  //            }
  //          } else if constexpr(std::is_same_v<T, boss::Span<double>>) {
  //            for(const auto& value : arg) {
  //              std::cout << value << " ";
  //            }
  //          } else {
  //            std::cout << "Unrecognised value ";
  //          }
  //        },
  //        oldSpans[0]);
  //  }
  //  std::cout << std::endl;
  //  std::cout << "Size of static argument types: " << sizeof...(StaticArgumentTypes) << std::endl;

  for(auto it = std::move_iterator(dynamics.begin()); it != std::move_iterator(dynamics.end());
      ++it) {
    std::visit(
        [&spanInputs]<typename InputType>(InputType&& argument) {
          using Type = std::decay_t<InputType>;
          if constexpr(boss::utilities::isVariantMember<std::vector<Type>, SpanInputs>::value) {
            if(spanInputs.size() > 0 &&
               std::holds_alternative<std::vector<Type>>(spanInputs.back())) {
              std::get<std::vector<Type>>(spanInputs.back()).push_back(argument);
            } else {
              spanInputs.push_back(std::vector<Type>{argument});
            }
          }
        },
        *it);
  }
  dynamics.erase(dynamics.begin(), dynamics.end());

  //  std::cout << "Dynamics size after transform: " << dynamics.size() << std::endl;
  //  std::cout << "Spans after transform: " << spanInputs.size() << std::endl;
  //  std::cout << "Span: ";
  //  std::visit([](auto&& arg) {
  //    using T = std::decay_t<decltype(arg)>;
  //    if constexpr (std::is_same_v<T, std::vector<long>>) {
  //      for (const auto& value : arg) {
  //        std::cout << value << " ";
  //      }
  //    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
  //      for (const auto& value : arg) {
  //        std::cout << value << " ";
  //      }
  //    } else {
  //      std::cout << "Unrecognised value ";
  //    }
  //  }, spanInputs[0]);
  //  std::cout << std::endl;

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

static boss::Expression toBOSSExpression(Expression&& expr) {
  return std::visit(boss::utilities::overload(
                        [&](ComplexExpression&& e) -> boss::Expression {
                          boss::ExpressionArguments bossArgs;
                          auto fromArgs = e.getArguments();
                          bossArgs.reserve(fromArgs.size());
                          std::transform(std::make_move_iterator(fromArgs.begin()),
                                         std::make_move_iterator(fromArgs.end()),
                                         std::back_inserter(bossArgs), [](auto&& bulkArg) {
                                           return toBOSSExpression(
                                               std::forward<decltype(bulkArg)>(bulkArg));
                                         });
                          return boss::ComplexExpression(e.getHead(), std::move(bossArgs));
                        },
                        [&](Pred&& e) -> boss::Expression {
                          boss::Expression output = static_cast<boss::Expression>(std::move(e));
                          return std::move(output);
                        },
                        [](auto&& otherTypes) -> boss::Expression { return otherTypes; }),
                    std::move(expr));
}

static Expression evaluateInternal(Expression&& e);

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
            [](auto&& span) -> ExpressionSpanArgument {
              return std::forward<decltype(span)>(span).subspan(1);
            },
            std::move(spans[0]));
      }

      return std::visit(
          [head = std::move(head), statics = std::move(statics), dynamics = std::move(dynamics),
           spans = std::move(spans), this](auto&& argument) mutable -> std::pair<Expression, bool> {
            typedef std::decay_t<decltype(argument)> ArgType;

            if constexpr(std::is_same_v<ArgType, typename std::tuple_element<
                                                     sizeof...(T), std::tuple<Args...>>::type>) {
              // argument type matching, add one more static argument to the expression
              return dispatchAndEvaluate(ComplexExpressionWithStaticArguments<T..., ArgType>(
                  head, std::tuple_cat(std::move(statics), std::make_tuple(std::move(argument))),
                  std::move(dynamics), std::move(spans)));
            } else {
              ExpressionArguments rest{};
              rest.emplace_back(std::move(argument));
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
      if(dynamics.size() > sizeof...(Args)) {
        std::transform(
            std::move_iterator(next(dynamics.begin(), sizeof...(Args))),
            std::move_iterator(dynamics.end()), std::back_inserter(rest),
            [](auto&& arg) { return evaluateInternal(std::forward<decltype(arg)>(arg)); });
      }
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

    (*this)["Multiply"_] =
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
    (*this)["Multiply"_] = [](ComplexExpressionWithStaticArguments<Symbol>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::multiplies());
    };
    (*this)["Multiply"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      return createLambdaExpression(std::move(input), std::multiplies());
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

    (*this)["Where"_] = [](ComplexExpressionWithStaticArguments<Pred>&& input) -> Expression {
      assert(input.getSpanArguments().empty());
      assert(input.getDynamicArguments().empty());
      return Pred(get<0>(std::move(input).getStaticArguments()));
    };

    (*this)["As"_] = [](ComplexExpression&& input) -> Expression { return std::move(input); };

    (*this)["Select"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = boss::get<ComplexExpression>(std::move(*it));
      auto predFunc = boss::get<Pred>(std::move(*++it));

      auto columns = std::move(relation).getDynamicArguments();
      std::transform(
          std::make_move_iterator(columns.begin()), std::make_move_iterator(columns.end()),
          columns.begin(), [](auto&& columnExpr) {
            auto column = get<ComplexExpression>(std::forward<decltype(columnExpr)>(columnExpr));
            auto [head, unused_, dynamics, spans] = std::move(column).decompose();
            //          std::cout << "Head of dynamics to transform to span: " << head << std::endl;
            auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
            //          std::cout << "List size: " << list.getDynamicArguments().size() <<
            //          std::endl;
            if(list.getDynamicArguments().size() > 0) {
              list = transformDynamicsToSpans(std::move(list));
            }
            dynamics.at(0) = std::move(list);
            return ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
          });

      while(auto predicate = predFunc(columns)) {
        std::transform(
            std::make_move_iterator(columns.begin()), std::make_move_iterator(columns.end()),
            columns.begin(), [&predicate](auto&& columnExpr) {
              auto column = get<ComplexExpression>(std::forward<decltype(columnExpr)>(columnExpr));
              auto [head, unused_, dynamics, spans] = std::move(column).decompose();
              auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
              auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
              listSpans.at(0) = std::visit(
                  [&predicate]<typename T>(Span<T>&& typedSpan) -> ExpressionSpanArgument {
                    if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t>) {
                      auto result = std::vector<std::decay_t<T>>(); // TODO - filter in place
                      //                      auto& indexes = std::get<Span<long>>(*predicate);
                      //                      for(auto& index : indexes) {
                      //                        result.push_back(typedSpan[index]);
                      //                      }
                      // TODO - this should be indexes for Select (would need to change bool below)
                      auto& qualified = std::get<Span<bool>>(*predicate);
                      for(size_t i = 0; i < qualified.size(); ++i) {
                        if(qualified[i]) {
                          result.push_back(typedSpan[i]);
                        }
                      }
                      return Span<T>(std::move(std::vector(result)));
                    } else {
                      throw std::runtime_error("unsupported column type in select");
                    }
                  },
                  std::move(listSpans.at(0)));
              dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                                 std::move(listSpans));
              return ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
            });
        break;
      }
      return ComplexExpression("Table"_, std::move(columns));

      /*auto [headPred, unusedPred_, dynamicsPred, unused2Pred_] = std::move(predExpr).decompose();
      if(headPred.getName() != "Where") {
        throw std::runtime_error("Select statement not followed by 'Where', instead: " +
                                 headPred.getName());
      }
      auto where = get<ComplexExpression>(std::move(dynamicsPred.at(0)));
      auto& op = where.getHead();
      auto whereColumn = get<Symbol>(where.getDynamicArguments().at(0));
      auto whereValue = get<int64_t>(where.getDynamicArguments().at(1));

      auto predFunc = [whereValue, &op]<typename T2>(T2 value) -> bool {
        if(op.getName() == "Greater") {
          return value > whereValue;
        } else if(op.getName() == "Equal") {
          return value == whereValue;
        } else {
          throw std::runtime_error("Unsupported select operator: " + op.getName());
        }
      };
      auto indexes = std::vector<int32_t>();

      for(auto& columnExpr : columns) {
        if(std::get<ComplexExpression>(columnExpr).getHead() == whereColumn) {
          const auto& column = std::get<ComplexExpression>(columnExpr).getDynamicArguments().at(0);
          std::visit(
              [&predFunc, &indexes](auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, Span<long>> ||
                             std::is_same_v<T, Span<double>>) {
                  for(int32_t i = 0; i < arg.size(); i++) {
                    if(predFunc(arg[i])) {
                      indexes.push_back(i);
                    }
                  }
                } else {
                  throw std::runtime_error("Unrecognised type");
                }
              },
              std::get<ComplexExpression>(column).getSpanArguments().at(0));
          break;
        }
      }

      std::transform(
          std::make_move_iterator(columns.begin()), std::make_move_iterator(columns.end()),
          columns.begin(), [&indexes](auto&& columnExpr) {
            auto column = get<ComplexExpression>(std::forward<decltype(columnExpr)>(columnExpr));
            auto [head, unused_, dynamics, spans] = std::move(column).decompose();
            //          std::cout << "Column to run SELECT on: " << head << std::endl;
            auto list = get<ComplexExpression>(std::move(dynamics.at(0)));
            auto [listHead, listUnused_, listDynamics, listSpans] = std::move(list).decompose();
            listSpans.at(0) = std::visit(
                [&indexes]<typename T>(Span<T>&& typedSpan) -> ExpressionSpanArgument {
                  if constexpr(std::is_same_v<T, int64_t>) {
                    auto result = std::vector<std::decay_t<T>>();
                    for(auto& index : indexes) {
                      result.push_back(typedSpan[index]);
                    }
                    return Span<T>(std::move(std::vector(result)));
                  } else {
                    throw std::runtime_error("unsupported column type in select");
                  }
                },
                std::move(listSpans.at(0)));

            dynamics.at(0) = ComplexExpression(std::move(listHead), {}, std::move(listDynamics),
                                               std::move(listSpans));
            return ComplexExpression(std::move(head), {}, std::move(dynamics), std::move(spans));
          });

      return ComplexExpression("Table"_, std::move(columns));*/
    };

    (*this)["Project"_] = [](ComplexExpression&& inputExpr) -> Expression {
      ExpressionArguments args = std::move(inputExpr).getArguments();
      auto it = std::make_move_iterator(args.begin());
      auto relation = boss::get<ComplexExpression>(std::move(*it));
      auto asExpr = std::move(*++it);
      if(relation.getHead().getName() != "Table") {
        return "Project"_(std::move(relation), std::move(asExpr));
      }

      // TODO - repeat of function in Select - factor out
      auto columns = std::move(relation).getDynamicArguments();
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

      auto projectedColumns = ExpressionArguments{};
      ExpressionArguments asArgs = boss::get<ComplexExpression>(std::move(asExpr)).getArguments();
      for(auto asIt = std::make_move_iterator(asArgs.begin());
          asIt != std::make_move_iterator(asArgs.end()); ++asIt) {
        auto name = get<Symbol>(std::move(*asIt));
        auto as = std::move(*++asIt);
        if(std::holds_alternative<Symbol>(as)) {
          auto asSymbol = get<Symbol>(std::move(as));
          for(auto& columnExpr : columns) {
            if(std::get<ComplexExpression>(columnExpr).getHead() == asSymbol) {
              auto& column = std::get<ComplexExpression>(columnExpr);
              auto [head, statics, dynamics, spans] = std::move(column).decompose();
              projectedColumns.emplace_back(ComplexExpression(
                  std::move(name), std::move(statics), std::move(dynamics), std::move(spans)));
              break;
            }
          }
        } else {
          auto asPred = get<Pred>(std::move(as));
          ExpressionSpanArguments spans{};
          while(auto projected = asPred(columns)) {
            std::visit([&spans](auto&& typedSpan) { spans.emplace_back(std::move(typedSpan)); },
                       *projected);
            break;
          }
          auto dynamics = ExpressionArguments{};
          dynamics.emplace_back(ComplexExpression("List"_, {}, {}, std::move(spans)));
          projectedColumns.emplace_back(
              ComplexExpression(std::move(name), {}, std::move(dynamics), {}));
        }
      }
      return ComplexExpression("Table"_, std::move(projectedColumns));
    };
  }

private:
  // Q: To create ComplexExpressionWithStaticArguments you must always explicitly call the
  // constructor with the associated types?
  template <typename T, typename F>
  static Pred createLambdaExpression(ComplexExpressionWithStaticArguments<T>&& e, F&& f) {
    assert(e.getSpanArguments().empty());
    assert(e.getDynamicArguments().size() == 1);
    Pred::Function pred = std::visit(
        [&e, &f](auto&& arg) -> Pred::Function {
          return
              [pred1 = createLambdaArgument(get<0>(e.getStaticArguments())),
               pred2 = createLambdaArgument(arg),
               f](ExpressionArguments& columns) mutable -> std::optional<ExpressionSpanArgument> {
                auto arg1 = pred1(columns);
                auto arg2 = pred2(columns);
                if(!arg1 || !arg2) {
                  return {};
                }
                ExpressionSpanArgument span;
                // Q: How do I construct values directly into a vector in a new Span? Or should we
                // create values in a vector and move this into a Span as below?
                std::visit( // TODO - this should be indexes for Select
                    [&span, f](auto&& typedSpan1, auto&& typedSpan2) {
                      using Type1 = std::decay_t<decltype(typedSpan1)>;
                      using Type2 = std::decay_t<decltype(typedSpan2)>;
                      if constexpr(std::is_same_v<Type1, Span<int64_t>> &&
                                   std::is_same_v<Type2, Span<int64_t>>) {
                        assert(typedSpan1.size() == 1 || typedSpan2.size() == 1);
                        using ElementType1 = typename Type1::element_type;
                        using ElementType2 = typename Type2::element_type;
                        using OutputType =
                            typename std::result_of<decltype(f)(ElementType1, ElementType2)>::type;
                        std::vector<OutputType> results;
                        if(typedSpan2.size() == 1) {
                          for(size_t i = 0; i < typedSpan1.size(); ++i) {
                            results.push_back(f(typedSpan1[i], typedSpan2[0]));
                          }
                        } else {
                          for(size_t i = 0; i < typedSpan2.size(); ++i) {
                            results.push_back(f(typedSpan1[0], typedSpan2[i]));
                          }
                        }
                        span = Span<OutputType>(std::move(std::vector(results)));
                      } else {
                        throw std::runtime_error("unsupported column type in select");
                      }
                    },
                    std::move(*arg1), std::move(*arg2));

                return span;
              };
        },
        // Q: Could this be a single funtion call, getDynamicArgumentAt(0)?
        e.getDynamicArguments().at(0));
    return {std::move(pred), toBOSSExpression(std::move(e))};
  }

  template <typename ArgType> static auto createLambdaArgument(ArgType const& arg) {
    if constexpr(std::is_same_v<ArgType, Symbol>) {
      return [arg](ExpressionArguments& columns) mutable -> std::optional<ExpressionSpanArgument> {
        // search for column matching the symbol in the relation
        for(auto& columnExpr : columns) {
          auto& column = get<ComplexExpression>(columnExpr);
          if(column.getHead() == arg) {
            auto& span =
                get<ComplexExpression>(column.getArguments().at(0)).getSpanArguments().at(0);
                // Q: When is the use of get required with the BOSS API
                // Q: Use of boss::get vs std::get
            return std::visit(
                []<typename T>(Span<T> const& typedSpan) -> std::optional<ExpressionSpanArgument> {
                  if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double_t>) {
                    return typedSpan.clone(
                        boss::expressions::CloneReason::FOR_TESTING); // TODO - return reference?
                  } else {
                    throw std::runtime_error("unsupported column type in predicate");
                  }
                },
                span);
          }
        }
        throw std::runtime_error("in predicate: unknown symbol " + arg.getName() + "_");
      };
    } else if constexpr(NumericType<ArgType>) {
      return [arg](ExpressionArguments& /*unused*/) -> std::optional<ExpressionSpanArgument> {
        return Span<ArgType>(
            std::move(std::vector({arg}))); // TODO - hacky to return 1 element span (Pred to return optional Expression?)
      };
    } else if constexpr(std::is_same_v<ArgType, Pred>) {
      return [f = static_cast<Pred::Function const&>(arg)](ExpressionArguments& columns) {
        return f(columns);
      };
    } else {
      throw std::runtime_error("unsupported argument type in predicate");
      return [](ExpressionArguments& /*unused*/) -> std::optional<ExpressionSpanArgument> {
        return {};
      };
    }
  }
};

static Expression evaluateInternal(Expression&& e) {
  static OperatorMap operators;
  return visit(boss::utilities::overload(
                   [](ComplexExpression&& e) -> Expression {
                     auto head = e.getHead();
                     if(operators.count(head))
                       return operators.at(head)(std::move(e));
                     return std::move(e);
                   },
                   [](auto&& e) -> Expression { return std::forward<decltype(e)>(e); }),
               std::move(e));
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
};

extern "C" void reset() {}