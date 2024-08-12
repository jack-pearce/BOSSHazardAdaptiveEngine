#ifndef BOSSHAZARDADAPTIVEENGINE_SHAREDDATATYPES_H
#define BOSSHAZARDADAPTIVEENGINE_SHAREDDATATYPES_H

#include <iostream>
#include <vector>

namespace adaptive {

enum SelectImplementation { Branch_, Predication_ };

std::ostream& operator<<(std::ostream& s, const SelectImplementation& implementation);

struct SelectOperatorState {
  SelectImplementation activeOperator;
  int32_t consecutivePredications;
  int32_t tuplesProcessed;
  int32_t branchMispredictions;
  int32_t selected;
  int32_t tuplesUntilHazardCheck;

  SelectOperatorState();
};

using SelectOperatorStates = std::vector<SelectOperatorState>;

std::ostream& operator<<(std::ostream& s, const SelectOperatorState& state);
std::ostream& operator<<(std::ostream& s, const SelectOperatorStates& states);

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SHAREDDATATYPES_H
