#include "operators/select.hpp"
#include "operators/partition.hpp"

namespace adaptive::config {

extern int32_t nonVectorizedDOP;
extern adaptive::Select selectImplementation;
extern adaptive::PartitionOperators partitionImplementation;
extern uint32_t minTuplesPerThread;

extern int32_t LOGICAL_CORE_COUNT;
extern std::string projectFilePath;


} // namespace adaptive::config