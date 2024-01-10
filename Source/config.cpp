#include "config.hpp"

namespace adaptive::config {

uint32_t DOP = 4;
adaptive::Select selectImplementation = adaptive::Select::AdaptiveParallel;

uint32_t minTuplesPerThread = 100;

} // namespace adaptive::config
