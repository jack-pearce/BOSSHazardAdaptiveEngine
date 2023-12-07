#ifndef BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP
#define BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP

#include <papi.h>
#include <string>
#include <vector>

namespace adaptive {

class Counters {
public:
  static Counters& getInstance();
  Counters(const Counters&) = delete;
  void operator=(const Counters&) = delete;

  long_long* getSharedEventSetEvents(std::vector<std::string>& counterNames);
  long_long* readSharedEventSet();

private:
  Counters();
  ~Counters();
  long_long* addEventsToSharedEventSet(std::vector<std::string>& counterNames);
  long_long* eventsAlreadyInSharedEventSet(std::vector<std::string>& counterNames);

  int sharedEventSet;
  std::vector<std::string> counters;
  long_long counterValues[20] = {0};
};

void createThreadEventSet(int* eventSet, std::vector<std::string>& counterNames);
void readThreadEventSet(int eventSet, int numEvents, long_long* values);
void destroyThreadEventSet(int eventSet, long_long* values);

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PAPIWRAPPER_HPP