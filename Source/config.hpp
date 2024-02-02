#include "operators/select.hpp"

namespace adaptive::config {

extern int32_t DOP;
extern adaptive::Select selectImplementation;

extern int32_t LOGICAL_CORE_COUNT;
extern std::string projectFilePath;
extern uint32_t minTuplesPerThread;

} // namespace adaptive::config