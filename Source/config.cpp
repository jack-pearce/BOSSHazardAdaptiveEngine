#include "config.hpp"

namespace adaptive::config {

int32_t nonVectorizedDOP = 4;
auto selectImplementation = adaptive::Select::Adaptive;
auto partitionImplementation = adaptive::PartitionOperators::RadixBitsAdaptiveParallel;
auto groupImplementation = adaptive::Group::GroupAdaptiveParallel;
uint32_t minTuplesPerThread = 100;

// TODO - machine config, these should be determined and set during the build process
int32_t LOGICAL_CORE_COUNT = 4;
// int32_t LOGICAL_CORE_COUNT = 10;
std::string projectFilePath = "/home/jcp122/repos/BOSSHazardAdaptiveEngine/";
// std::string projectFilePath = "/repos/BOSSHazardAdaptiveEngine/";

} // namespace adaptive::config
