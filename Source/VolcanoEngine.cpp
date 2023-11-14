
#include "VolcanoEngine.hpp"
#include "BOSSExpressionConversions.hpp"
#include "RelationalOps/Join.hpp"
#include "RelationalOps/Operator.hpp"
#include "RelationalOps/Project.hpp"
#include "RelationalOps/Relation.hpp"
#include "RelationalOps/Select.hpp"
#include "RelationalOps/Top.hpp"

#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>

#include <memory>
#include <mutex>

using boss::utilities::operator""_;
using boss::expressions::generic::isComplexExpression;

namespace boss::engines::volcano {

std::unique_ptr<operators::Operator> buildOperatorPipeline(ComplexExpression&& e) {
  if(e.getHead() == "Table"_) {
    auto [schema, data] = toSchemaAndData(std::move(e));
    return std::make_unique<operators::Relation>(std::move(schema), std::move(data));
  }
  if(e.getHead() == "Project"_) {
    auto [head, unused_, dynamics, unused2_] = std::move(e).decompose();
    auto it = std::make_move_iterator(dynamics.begin());
    auto itEnd = std::make_move_iterator(dynamics.end());
    auto input = buildOperatorPipeline(boss::get<ComplexExpression>(std::move(*it++)));
    auto [schema, projection] = toSchemaAndProjection(std::move(it), std::move(itEnd), *input);
    return std::make_unique<operators::Project>(std::move(input), std::move(schema),
                                                std::move(projection));
  }
  if(e.getHead() == "Select"_) {
    auto [head, unused_, dynamics, unused2_] = std::move(e).decompose();
    auto it = std::make_move_iterator(dynamics.begin());
    auto input = buildOperatorPipeline(boss::get<ComplexExpression>(std::move(*it++)));
    auto predExpr = boss::get<ComplexExpression>(std::move(*it++));
    return std::make_unique<operators::Select>(std::move(input), std::move(predExpr));
  }
  if(e.getHead() == "Join"_) {
    auto [head, unused_, dynamics, unused2_] = std::move(e).decompose();
    auto it = std::make_move_iterator(dynamics.begin());
    auto leftSideInput = buildOperatorPipeline(boss::get<ComplexExpression>(std::move(*it++)));
    auto rightSideInput = buildOperatorPipeline(boss::get<ComplexExpression>(std::move(*it++)));
    auto predExpr = boss::get<ComplexExpression>(std::move(*it++));
    return std::make_unique<operators::Join>(std::move(leftSideInput), std::move(rightSideInput),
                                             std::move(predExpr));
  }
  if(e.getHead() == "Top"_) {
    auto [head, unused_, dynamics, unused2_] = std::move(e).decompose();
    auto it = std::make_move_iterator(dynamics.begin());
    auto input = buildOperatorPipeline(boss::get<ComplexExpression>(std::move(*it++)));
    auto n = boss::get<int64_t>(std::move(*it++));
    auto orderExpr = boss::get<ComplexExpression>(std::move(*it++));
    return std::make_unique<operators::Top>(std::move(input), n, std::move(orderExpr));
  }
  throw std::runtime_error("Unknown relational operator: " + e.getHead().getName());
}

boss::Expression Engine::evaluate(Expression&& expr) { // NOLINT
  try {
    return visit(
        boss::utilities::overload(
            [this](ComplexExpression&& e) -> Expression {
              // convert the query expression into a volcano pipeline
              auto relationalOp = buildOperatorPipeline(std::move(e));
              std::vector<ComplexExpression> exprs;
              // process the tuples, decompose and insert into list expressions
              while(exprs.size() < relationalOp->getSchema().size()) {
                exprs.emplace_back("List"_());
              }
              while(auto tuple = relationalOp->next()) {
                auto it = exprs.begin();
                for(auto&& val : *tuple) {
                  auto& expr = *it++;
                  auto [head, unused_, dynamics, unused2_] = std::move(expr).decompose();
                  auto dyn = std::move(dynamics); // workaround because of the capture below
                  std::visit([&dyn](auto&& typedVal) { dyn.emplace_back(typedVal); },
                             std::move(val));
                  expr = ComplexExpression(std::move(head), std::move(dyn));
                }
              }
              // wrap the list expressions into a table expression
              ExpressionArguments args;
              for(auto&& expr : exprs) {
                ExpressionArguments columnArgs;
                columnArgs.emplace_back(std::move(expr));
                args.emplace_back(ComplexExpression(Symbol{relationalOp->getSchema()[args.size()]},
                                                    std::move(columnArgs)));
              }
              return ComplexExpression("Table"_, std::move(args));
            },
            [this](Symbol&& symbol) -> Expression { return std::move(symbol); },
            [](auto&& arg) -> Expression { return std::forward<decltype(arg)>(arg); }),
        std::move(expr));
  } catch(std::exception const& e) {
    ExpressionArguments args;
    args.emplace_back(std::move(expr));
    args.emplace_back(std::string{e.what()});
    return ComplexExpression{"ErrorWhenEvaluatingExpression"_, std::move(args)};
  }
}

} // namespace boss::engines::volcano

static auto& enginePtr(bool initialise = true) {
  static auto engine = std::unique_ptr<boss::engines::volcano::Engine>();
  if(!engine && initialise) {
    engine.reset(new boss::engines::volcano::Engine());
  }
  return engine;
}

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  static std::mutex m;
  std::lock_guard lock(m);
  auto* r = new BOSSExpression{enginePtr()->evaluate(std::move(e->delegate))};
  return r;
};

extern "C" void reset() { enginePtr(false).reset(nullptr); }
