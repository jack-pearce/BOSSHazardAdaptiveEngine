#pragma once

#include "Operator.hpp"
#include <vector>

namespace boss::engines::volcano::operators {

class Relation : public Operator {
public:
  Relation(Schema&& s, std::vector<Tuple>&& d)
      : schema(std::move(s)), data(std::move(d)), dataIt(data.begin()) {}

  std::optional<Tuple> next() override {
    if(dataIt != data.end()) {
      return *dataIt++;
    }
    return {};
  }

  Schema const& getSchema() const override { return schema; }

private:
  Schema schema;
  std::vector<Tuple> data;
  std::vector<Tuple>::iterator dataIt;
};

} // namespace boss::engines::volcano::operators