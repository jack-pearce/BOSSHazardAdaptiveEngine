#include <algorithm>
#include <cassert>
#include <iostream>
#include <pthread.h>

#include "papiWrapper.hpp"

namespace adaptive {

Counters& Counters::getInstance() {
  static Counters instance;
  return instance;
}

Counters::Counters() {
  sharedEventSet = PAPI_NULL;
  std::vector<std::string> initialCounters = {"PERF_COUNT_HW_CPU_CYCLES"};

  if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    std::cerr << "PAPI library init error!" << std::endl;
    exit(1);
  }

  if(PAPI_thread_init(pthread_self) != PAPI_OK) {
    std::cerr << "PAPI thread init error!" << std::endl;
    exit(1);
  }

  if(PAPI_create_eventset(&sharedEventSet) != PAPI_OK) {
    std::cerr << "PAPI could not create event set!" << std::endl;
    exit(1);
  }

  addEventsToSharedEventSet(initialCounters);
  PAPI_start(sharedEventSet);
}

Counters::~Counters() {
  PAPI_stop(sharedEventSet, counterValues);
  PAPI_cleanup_eventset(sharedEventSet);
  PAPI_destroy_eventset(&sharedEventSet);
  PAPI_shutdown();
}

long_long* Counters::addEventsToSharedEventSet(std::vector<std::string>& counterNames) {
  int eventCode;
  PAPI_stop(sharedEventSet, counterValues);

  for(const std::string& counter : counterNames) {
    if(__builtin_expect(PAPI_event_name_to_code(counter.c_str(), &eventCode) != PAPI_OK, false)) {
      std::cerr << "PAPI could not create event code!" << std::endl;
      exit(1);
    }

    if(__builtin_expect(PAPI_add_event(sharedEventSet, eventCode) != PAPI_OK, 0)) {
      std::cerr << "Could not add '" << counter << "' to event set!" << std::endl;
      exit(1);
    }
  }

  counters.insert(counters.end(), counterNames.begin(), counterNames.end());
  PAPI_start(sharedEventSet);
  return &(counterValues[counters.size() - counterNames.size()]);
}

long_long* Counters::eventsAlreadyInSharedEventSet(std::vector<std::string>& counterNames) {
  for(size_t i = 0; i <= counters.size(); ++i) {
    if(counters[i] == counterNames[0]) {
      bool found = true;

      for(size_t j = 1; j < counterNames.size(); ++j) {
        if(counters[i + j] != counterNames[j]) {
          found = false;
          break;
        }
      }

      if(found) {
        return &(counterValues[i]);
      }
    }
  }
  return nullptr;
}

long_long* Counters::getSharedEventSetEvents(std::vector<std::string>& counterNames) {
  long_long* eventValues = eventsAlreadyInSharedEventSet(counterNames);
  if(__builtin_expect(eventValues != nullptr, true))
    return eventValues;

  return addEventsToSharedEventSet(counterNames);
}

long_long* Counters::readSharedEventSet() {
  for(size_t i = 1; i < counters.size(); ++i) {
    counterValues[i] = 0;
  }

  if(__builtin_expect(PAPI_accum(sharedEventSet, counterValues) != PAPI_OK, false)) {
    std::cerr << "Could not read and zero event set!" << std::endl;
    exit(1);
  }
  return counterValues;
}

void createThreadEventSet(int* eventSet, std::vector<std::string>& counterNames) {
  assert(*eventSet == PAPI_NULL);
  if(__builtin_expect(PAPI_create_eventset(eventSet) != PAPI_OK, false)) {
    std::cerr << "Could not create additional event set!" << std::endl;
    exit(1);
  }

  int eventCode;
  for(const std::string& counter : counterNames) {
    if(__builtin_expect(PAPI_event_name_to_code(counter.c_str(), &eventCode) != PAPI_OK, false)) {
      std::cerr << "PAPI could not create event code!" << std::endl;
      exit(1);
    }

    if(__builtin_expect(PAPI_add_event(*eventSet, eventCode) != PAPI_OK, 0)) {
      std::cerr << "Could not add '" << counter << "' to event set!" << std::endl;
      exit(1);
    }
  }

  PAPI_start(*eventSet);
}

void readThreadEventSet(int eventSet, int numEvents, long_long* values) {
  for(int i = 0; i < numEvents; i++) {
    *(values + i) = 0;
  }

  if(__builtin_expect(PAPI_accum(eventSet, values) != PAPI_OK, false)) {
    std::cerr << "Could not read and zero event set!" << std::endl;
    exit(1);
  }
}

void destroyThreadEventSet(int eventSet, long_long* values) {
  PAPI_stop(eventSet, values);
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
}

} // namespace adaptive