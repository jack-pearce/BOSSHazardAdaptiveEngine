#include <algorithm>
#include <iostream>
#include <pthread.h>

#include "papiWrapper.hpp"

// #define DEBUG

namespace adaptive {

void initialisePapi() {
  if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    std::cerr << "PAPI library init error!" << std::endl;
    exit(1);
  }

  if(PAPI_thread_init(pthread_self) != PAPI_OK) {
    std::cerr << "PAPI thread init error!" << std::endl;
    exit(1);
  }
#ifdef DEBUG
  std::cout << "PAPI library initialised" << std::endl;
#endif
}

void shutdownPapi() {
  PAPI_shutdown();
#ifdef DEBUG
  std::cout << "PAPI library shutdown" << std::endl;
#endif
}

PAPI_eventSet::PAPI_eventSet(const std::vector<std::string>& counterNames)
    : eventSet(PAPI_NULL), counterValues(counterNames.size(), 0),
      counterValuesDiff(counterNames.size(), 0) {
  if(PAPI_create_eventset(&eventSet) != PAPI_OK) {
    std::cerr << "PAPI could not create event set!" << std::endl;
    exit(1);
  }

  int eventCode;
  for(const std::string& counterName : counterNames) {
    if(__builtin_expect(PAPI_event_name_to_code(counterName.c_str(), &eventCode) != PAPI_OK,
                        false)) {
      std::cerr << "PAPI could not create event code!" << std::endl;
      exit(1);
    }

    if(__builtin_expect(PAPI_add_event(eventSet, eventCode) != PAPI_OK, 0)) {
      std::cerr << "Could not add '" << counterName << "' to event set!" << std::endl;
      exit(1);
    }
  }

  PAPI_start(eventSet);
#ifdef DEBUG
  std::cout << "PAPI event set created" << std::endl;
#endif
}

PAPI_eventSet::~PAPI_eventSet() {
  PAPI_stop(eventSet, counterValues.data());

  if(PAPI_cleanup_eventset(eventSet) != PAPI_OK) {
    std::cerr << "PAPI could not clean up create event set!" << std::endl;
    exit(1);
  }

  if(PAPI_destroy_eventset(&eventSet) != PAPI_OK) {
    std::cerr << "PAPI could not destroy event set!" << std::endl;
    exit(1);
  }
#ifdef DEBUG
  std::cout << "PAPI event set destroyed" << std::endl;
#endif
}

long_long* PAPI_eventSet::getCounterDiffsPtr() { return counterValuesDiff.data(); }

void PAPI_eventSet::readCounters() {
  if(__builtin_expect(PAPI_read(eventSet, counterValues.data()) != PAPI_OK, false)) {
    std::cerr << "Could not read event set!" << std::endl;
    exit(1);
  }
}

void PAPI_eventSet::readCountersAndUpdateDiff() {
  std::copy(counterValues.begin(), counterValues.end(), counterValuesDiff.begin());

  if(__builtin_expect(PAPI_read(eventSet, counterValues.data()) != PAPI_OK, false)) {
    std::cerr << "Could not read event set!" << std::endl;
    exit(1);
  }

  for(size_t i = 0; i < counterValuesDiff.size(); ++i) {
    counterValuesDiff[i] = counterValues[i] - counterValuesDiff[i];
  }
}

PAPI_eventSet& getThreadEventSet() {
  thread_local static PAPI_eventSet eventSet({"PERF_COUNT_HW_CPU_CYCLES",
                                              "PERF_COUNT_HW_BRANCH_MISSES", "DTLB-STORE-MISSES",
                                              "DTLB-LOAD-MISSES", "PERF_COUNT_HW_CACHE_MISSES"});
  return eventSet;
}

} // namespace adaptive