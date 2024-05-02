#ifndef BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP
#define BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP

#include <papi.h>
#include <string>
#include <vector>

namespace adaptive {

void initialisePapi();
void shutdownPapi();

class PAPI_eventSet {
public:
  PAPI_eventSet(const PAPI_eventSet&) = delete;
  void operator=(const PAPI_eventSet&) = delete;

  explicit PAPI_eventSet(const std::vector<std::string>& counterNames);
  ~PAPI_eventSet();
  long_long* getCounterDiffsPtr();
  long_long* getCounterValuesPtr();
  void readCounters();
  void readCountersAndUpdateDiff();
  [[nodiscard]] int getEventSet() const;

private:
  int eventSet;
  std::vector<long_long> counterValues;
  std::vector<long_long> counterValuesDiff;
};

PAPI_eventSet& getThreadEventSet();
void switchEventSetToCycles(PAPI_eventSet& eventSet);
void switchEventSetToPartition(PAPI_eventSet& eventSet);

// Must match the actual ThreadEventSet in papiWrapper.cpp
enum EVENT {
  PERF_COUNT_HW_BRANCH_MISSES = 0,
  DTLB_LOAD_MISSES = 1,
  PERF_COUNT_HW_CACHE_MISSES = 2,
  DTLB_STORE_MISSES = 3,
  PERF_COUNT_HW_CPU_CYCLES = 3
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP