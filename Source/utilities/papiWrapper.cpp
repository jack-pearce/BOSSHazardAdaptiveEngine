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
  eventSet = PAPI_NULL;
  auto counterNames =
      std::vector<std::string>({"PERF_COUNT_HW_CPU_CYCLES", "PERF_COUNT_HW_BRANCH_MISSES"});

  if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    std::cerr << "PAPI library init error!" << std::endl;
    exit(1);
  }

  if(PAPI_thread_init(pthread_self) != PAPI_OK) {
    std::cerr << "PAPI thread init error!" << std::endl;
    exit(1);
  }

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
}

Counters::~Counters() {
  PAPI_stop(eventSet, counterValues);
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
  PAPI_shutdown();
}

long_long* Counters::getBranchMisPredictionsCounter() { return &(counterValuesDiff[1]); }

long_long* Counters::readEventSetAndGetCycles() {
  if(__builtin_expect(PAPI_read(eventSet, counterValues) != PAPI_OK, false)) {
    std::cerr << "Could not read and zero event set!" << std::endl;
    exit(1);
  }
  return &(counterValuesDiff[0]);
}

void Counters::readEventSetAndCalculateDiff() {
  for(auto i = 1; i < COUNTERS; ++i) {
    counterValuesDiff[i] = counterValues[i];
  }

  if(__builtin_expect(PAPI_read(eventSet, counterValues) != PAPI_OK, false)) {
    std::cerr << "Could not read and zero event set!" << std::endl;
    exit(1);
  }

  for(auto i = 1; i < COUNTERS; ++i) {
    counterValuesDiff[i] = counterValues[i] - counterValuesDiff[i];
  }
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