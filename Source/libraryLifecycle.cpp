#include "utilities/papiWrapper.hpp"
#include "constants/machineConstants.hpp"
#include <iostream>

// #define DEBUG

void init() __attribute__((constructor));
void cleanup() __attribute__((destructor));

void init() {
  adaptive::initialisePapi();
  adaptive::MachineConstants::getInstance(); // Will calculate any missing machine constants
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