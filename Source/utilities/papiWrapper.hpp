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
  void readCounters();
  void readCountersAndUpdateDiff();

private:
  int eventSet;
  std::vector<long_long> counterValues;
  std::vector<long_long> counterValuesDiff;
};

PAPI_eventSet& getThreadEventSet();

// Must match the actual ThreadEventSet in papiWrapper.cpp
enum EVENT {
  PERF_COUNT_HW_CPU_CYCLES,
  PERF_COUNT_HW_BRANCH_MISSES,
  DTLB_STORE_MISSES,
  DTLB_LOAD_MISSES,
  PERF_COUNT_HW_CACHE_MISSES
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP