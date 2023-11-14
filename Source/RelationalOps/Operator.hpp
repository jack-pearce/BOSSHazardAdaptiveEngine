#pragma once

#include "../Types.hpp"
#include <optional>

namespace boss::engines::volcano::operators {

class Operator {
public:
  virtual std::optional<Tuple> next() = 0;

  Operator() = default;          // acts as open()
  virtual ~Operator() = default; // acts as close()

  // not strictly belonging here,
  // but convenient for getting the schema changes along the pipeline (i.e., projections and joins)
  virtual Schema const& getSchema() const = 0;
};

} // namespace boss::engines::volcano::operators
