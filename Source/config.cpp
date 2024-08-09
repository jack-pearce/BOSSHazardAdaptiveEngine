#include "config.hpp"

namespace adaptive::config {

// NOLINTBEGIN
int32_t nonVectorizedDOP = 1;
auto selectImplementation = adaptive::Select::Adaptive;
auto partitionImplementation = adaptive::PartitionOperators::RadixBitsAdaptiveParallel;
auto groupImplementation = adaptive::Group::GroupAdaptiveParallel;
bool DEFER_GATHER_PHASE_OF_SELECT_TO_OTHER_ENGINES = false;
bool CONSTANTS_INITIALISED = false; // TODO to remove (make part of installation)
int minPartitionSize = 300000;
// NOLINTEND

const uint32_t minTuplesPerThread = 100;

// TODO - machine config, these should be determined and set during the build process
// const int32_t LOGICAL_CORE_COUNT = 4;
const int32_t LOGICAL_CORE_COUNT = 10;
const std::string projectFilePath = "/adaptive/BOSSHazardAdaptiveEngine/";
// const std::string projectFilePath = "/repos/BOSSHazardAdaptiveEngine/";

} // namespace adaptive::config
