#ifndef BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP
#define BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP

#include <cstdint>
#include <string>

namespace adaptive {

enum Select { Branch, Predication, Adaptive, AdaptiveParallel };
enum PartitionOperators { RadixBitsFixed, RadixBitsAdaptive, RadixBitsAdaptiveParallel };

namespace config {

extern int32_t nonVectorizedDOP;
extern adaptive::Select selectImplementation;
extern adaptive::PartitionOperators partitionImplementation;
extern uint32_t minTuplesPerThread;

extern int32_t LOGICAL_CORE_COUNT;
extern std::string projectFilePath;

} // namespace config

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP