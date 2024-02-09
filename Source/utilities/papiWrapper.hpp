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

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP