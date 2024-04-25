#include "utilities/papiWrapper.hpp"

// #define DEBUG

void init() __attribute__((constructor));
void cleanup() __attribute__((destructor));

void init() {
  adaptive::initialisePapi();
#ifdef DEBUG
  std::cout << "BOSSHazardAdaptiveEngine library initialized" << std::endl;
#endif
}

void cleanup() {
  adaptive::shutdownPapi();
#ifdef DEBUG
  std::cout << "BOSSHazardAdaptiveEngine library cleaned up" << std::endl;
#endif
}