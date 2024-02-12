#ifndef BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
#define BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H

#include "utilities/sharedDataTypes.hpp"

using adaptive::SelectOperatorState;
using adaptive::SelectOperatorStates;

namespace adaptive {

class EngineInstanceState {
public:
  EngineInstanceState() : selectOperatorStates(nullptr), vectorizedDOP(-1) {}

  void setStatsPtr(SelectOperatorStates* newPtr) { selectOperatorStates = newPtr; }
  [[nodiscard]] SelectOperatorState& getStateOfID(int id) const {
    return (*selectOperatorStates)[id];
  }

  void setVectorizedDOP(int32_t vectorizedDOP_) { vectorizedDOP = vectorizedDOP_; }
  [[nodiscard]] int32_t getVectorizedDOP() const { return vectorizedDOP; }

private:
  SelectOperatorStates* selectOperatorStates;
  int32_t vectorizedDOP;
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_OPERATORSTATS_H
