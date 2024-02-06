#ifndef BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
#define BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H

#include "utilities/sharedDataTypes.hpp"

using adaptive::SelectOperatorState;
using adaptive::SelectOperatorStates;

namespace adaptive {

class SelectOperatorStats {
private:
  SelectOperatorStates* selectOperatorStates;

  SelectOperatorStats() : selectOperatorStates(nullptr) {}

public:
  static SelectOperatorStats& getInstance() {
    static SelectOperatorStats instance;
    return instance;
  }

  void setStatsPtr(SelectOperatorStates* newPtr) { selectOperatorStates = newPtr; }

  [[nodiscard]] SelectOperatorStates* getStatsPtr() const { return selectOperatorStates; }

  [[nodiscard]] SelectOperatorState& getStateOfID(int id) const {
    return (*selectOperatorStates)[id];
  }
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
