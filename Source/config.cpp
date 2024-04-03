#include "config.hpp"

namespace adaptive::config {

int32_t nonVectorizedDOP = 1;
adaptive::Select selectImplementation = adaptive::Select::Adaptive;
adaptive::PartitionOperators partitionImplementation =
    adaptive::PartitionOperators::RadixBitsAdaptiveParallel;
adaptive::Group groupImplementation = adaptive::Group::Hash;
uint32_t minTuplesPerThread = 100;

// TODO - machine config, these should be determined and set during the build process
int32_t LOGICAL_CORE_COUNT = 4;
// int32_t LOGICAL_CORE_COUNT = 10;
std::string projectFilePath = "/home/jcp122/repos/BOSSHazardAdaptiveEngine/";
// std::string projectFilePath = "/repos/BOSSHazardAdaptiveEngine/";

} // namespace adaptive::config
