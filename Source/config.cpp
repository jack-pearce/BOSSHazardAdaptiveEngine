#include "config.hpp"

namespace adaptive::config {

uint32_t DOP = 1;
adaptive::Select selectImplementation = adaptive::Select::Branch;

uint32_t minTuplesPerThread = 100;

} // namespace adaptive::config
