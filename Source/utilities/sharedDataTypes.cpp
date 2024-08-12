#include "sharedDataTypes.hpp"

namespace adaptive {

std::ostream& operator<<(std::ostream& s, const SelectImplementation& implementation) {
  switch(implementation) {
  case SelectImplementation::Branch_:
    s << "Branch";
    break;
  case SelectImplementation::Predication_:
    s << "Predication";
    break;
  default:
    s << "Unknown active operator";
  }
  return s;
}

SelectOperatorState::SelectOperatorState()
    : activeOperator(SelectImplementation::Predication_), consecutivePredications(-1),
      tuplesProcessed(0), branchMispredictions(0), selected(0), tuplesUntilHazardCheck(-1) {}

std::ostream& operator<<(std::ostream& s, const SelectOperatorState& state) {
  s << "Active Operator: " << state.activeOperator << "\n"
    << "Consecutive Predication Batches: " << state.consecutivePredications << "\n"
    << "Tuples Processed: " << state.tuplesProcessed << "\n"
    << "Branch Mispredictions: " << state.branchMispredictions << "\n"
    << "Selected: " << state.selected << "\n"
    << "Tuples Until Hazard Check: " << state.tuplesUntilHazardCheck << "\n";
  return s;
}

std::ostream& operator<<(std::ostream& s, const SelectOperatorStates& states) {
  for(size_t i = 0; i < states.size(); ++i) {
    s << "SelectID = " << i << "\n";
    s << states[i];
  }
  return s;
}

} // namespace adaptive
