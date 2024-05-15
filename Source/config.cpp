#include "config.hpp"

namespace adaptive::config {

int32_t nonVectorizedDOP = 1;
auto selectImplementation = adaptive::Select::Adaptive;
auto partitionImplementation = adaptive::PartitionOperators::RadixBitsAdaptiveParallel;
auto groupImplementation = adaptive::Group::GroupAdaptiveParallel;
uint32_t minTuplesPerThread = 100;
int minPartitionSize = 100000;

bool DEFER_GATHER_PHASE_OF_SELECT_TO_OTHER_ENGINES = true;

// TODO - machine config, these should be determined and set during the build process
// int32_t LOGICAL_CORE_COUNT = 4;
int32_t LOGICAL_CORE_COUNT = 10;
std::string projectFilePath = "/adaptive/BOSSHazardAdaptiveEngine/";
// std::string projectFilePath = "/repos/BOSSHazardAdaptiveEngine/";

} // namespace adaptive::config
