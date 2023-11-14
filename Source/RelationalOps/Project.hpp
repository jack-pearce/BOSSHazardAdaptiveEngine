#pragma once

#include "Operator.hpp"
#include "../Types.hpp"
#include <memory>

namespace boss::engines::volcano::operators {

class Project : public Operator {
public:
  Project(std::unique_ptr<Operator>&& op, Schema&& schema, Projection&& proj)
      : input(std::move(op)), schema(std::move(schema)), projection(std::move(proj)) {}

  std::optional<Tuple> next() override {
    if(auto tuple = input->next()) {
      return projection(*tuple);
    }
    return {};
  }

  Schema const& getSchema() const override { return schema; }

private:
  std::unique_ptr<Operator> input;
  Projection projection;
  Schema schema; // new schema after projections
};

} // namespace boss::engines::volcano::operators
