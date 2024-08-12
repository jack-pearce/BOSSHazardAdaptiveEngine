#ifndef BOSSHAZARDADAPTIVEENGINE_HAZARDADAPTIVEENGINE_H
#define BOSSHAZARDADAPTIVEENGINE_HAZARDADAPTIVEENGINE_H

#include <BOSS.hpp>
#include <Expression.hpp>

class PredWrapper;

using HAExpressionSystem = boss::expressions::generic::ExtensibleExpressionSystem<PredWrapper>;
using AtomicExpression = HAExpressionSystem::AtomicExpression;
using ComplexExpression = HAExpressionSystem::ComplexExpression;
template <typename... T>
using ComplexExpressionWithStaticArguments =
    HAExpressionSystem::ComplexExpressionWithStaticArguments<T...>;
using Expression = HAExpressionSystem::Expression;
using ExpressionArguments = HAExpressionSystem::ExpressionArguments;
using ExpressionSpanArguments = HAExpressionSystem::ExpressionSpanArguments;
using ExpressionSpanArgument = HAExpressionSystem::ExpressionSpanArgument;
using boss::Span;

#endif // BOSSHAZARDADAPTIVEENGINE_HAZARDADAPTIVEENGINE_H
