#ifndef BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP
#define BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP

#include <papi.h>
#include <string>
#include <vector>

#define COUNTERS 2 // "PERF_COUNT_HW_CPU_CYCLES", "PERF_COUNT_HW_BRANCH_MISSES"

namespace adaptive {

class Counters {
public:
  static Counters& getInstance();
  Counters(const Counters&) = delete;
  void operator=(const Counters&) = delete;

  long_long* getBranchMisPredictionsCounter();
  long_long* readEventSetAndGetCycles();
  void readEventSetAndCalculateDiff();

private:
  Counters();
  ~Counters();

  int eventSet;
  long_long counterValues[COUNTERS] = {0};
  long_long counterValuesDiff[COUNTERS] = {0};
};

void createThreadEventSet(int* eventSet, std::vector<std::string>& counterNames);
void readThreadEventSet(int eventSet, int numEvents, long_long* values);
void destroyThreadEventSet(int eventSet, long_long* values);

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP