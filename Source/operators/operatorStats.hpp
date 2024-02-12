#ifndef BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
#define BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H

#include "utilities/sharedDataTypes.hpp"

using adaptive::SelectOperatorState;
using adaptive::SelectOperatorStates;

namespace adaptive {

class SelectOperatorStats {
public:
  SelectOperatorStats() : selectOperatorStates(nullptr) {}
  void setStatsPtr(SelectOperatorStates* newPtr) { selectOperatorStates = newPtr; }
  [[nodiscard]] SelectOperatorState& getStateOfID(int id) const {
    return (*selectOperatorStates)[id];
  }

private:
  SelectOperatorStates* selectOperatorStates;
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
