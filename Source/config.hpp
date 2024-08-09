#ifndef BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP
#define BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP

#include <cstdint>
#include <string>

namespace adaptive {

enum Select { Branch, Predication, Adaptive, AdaptiveParallel };
enum PartitionOperators {
  RadixBitsFixedMin,
  RadixBitsFixedMax,
  RadixBitsAdaptive,
  RadixBitsAdaptiveParallel
};
enum Group { Hash, Sort, GroupAdaptive, GroupAdaptiveParallel };

namespace config {

// NOLINTBEGIN
extern int32_t nonVectorizedDOP;
extern adaptive::Select selectImplementation;
extern adaptive::PartitionOperators partitionImplementation;
extern adaptive::Group groupImplementation;
extern bool DEFER_GATHER_PHASE_OF_SELECT_TO_OTHER_ENGINES;
extern bool CONSTANTS_INITIALISED;
extern int minPartitionSize;
// NOLINTEND

extern const uint32_t minTuplesPerThread;

extern const int32_t LOGICAL_CORE_COUNT;
extern const std::string projectFilePath;

} // namespace config

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_CONFIG_HPP
